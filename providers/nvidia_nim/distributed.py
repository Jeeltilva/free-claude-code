"""Distributed persistence for NVIDIA NIM account pool state."""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DistributedState:
    """Distributed state representation for account pool."""

    # Account health state
    account_states: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Pool-level metrics
    total_requests: int = 0
    total_failures: int = 0
    last_sync_time: float = 0.0

    # Version tracking for conflict resolution
    version: int = 0
    node_id: str = ""

    # Timestamp for consistency
    timestamp: float = 0.0


class DistributedPersistence:
    """Manages distributed state persistence for multi-node deployments."""

    def __init__(
        self,
        node_id: str,
        state_file: str = "",
        sync_interval: int = 30,
        backup_interval: int = 300,
    ):
        """Initialize distributed persistence.

        Args:
            node_id: Unique identifier for this node
            state_file: Path to local state file for persistence
            sync_interval: Seconds between sync attempts
            backup_interval: Seconds between backups
        """
        self.node_id = node_id
        self.state_file = state_file
        self.sync_interval = sync_interval
        self.backup_interval = backup_interval

        # In-memory state cache
        self._local_state = DistributedState(node_id=node_id)
        self._last_backup_time = 0.0
        self._backup_files: List[str] = []

        # Load existing state
        self._load_local_state()

    def _load_local_state(self) -> None:
        """Load state from local file if available."""
        if not self.state_file:
            return

        try:
            path = Path(self.state_file)
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._local_state = self._deserialize_state(data)
                logger.info(f"DIST: Loaded state from {self.state_file}")
        except Exception as e:
            logger.warning(f"DIST: Failed to load state: {e}")

    def _serialize_state(self, state: DistributedState) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "account_states": state.account_states,
            "total_requests": state.total_requests,
            "total_failures": state.total_failures,
            "last_sync_time": state.last_sync_time,
            "version": state.version,
            "node_id": state.node_id,
            "timestamp": state.timestamp,
        }

    def _deserialize_state(self, data: Dict[str, Any]) -> DistributedState:
        """Deserialize state from dictionary."""
        state = DistributedState()
        state.account_states = data.get("account_states", {})
        state.total_requests = data.get("total_requests", 0)
        state.total_failures = data.get("total_failures", 0)
        state.last_sync_time = data.get("last_sync_time", 0.0)
        state.version = data.get("version", 0)
        state.node_id = data.get("node_id", "")
        state.timestamp = data.get("timestamp", 0.0)
        return state

    def update_account_state(self, account_index: int, account_data: Dict[str, Any]) -> None:
        """Update account state in local cache.

        Args:
            account_index: Account index
            account_data: Account state data
        """
        self._local_state.account_states[account_index] = account_data
        self._local_state.version += 1
        self._local_state.timestamp = time.time()

    def increment_pool_metrics(self, requests: int = 0, failures: int = 0) -> None:
        """Increment pool-level metrics.

        Args:
            requests: Number of requests to increment
            failures: Number of failures to increment
        """
        self._local_state.total_requests += requests
        self._local_state.total_failures += failures
        self._local_state.version += 1
        self._local_state.timestamp = time.time()

    def get_local_state(self) -> DistributedState:
        """Get current local state."""
        # Update timestamp before returning
        self._local_state.timestamp = time.time()
        return self._local_state

    def merge_remote_state(self, remote_state: DistributedState) -> bool:
        """Merge remote state with local state using version-based conflict resolution.

        Args:
            remote_state: Remote state to merge

        Returns:
            True if state was updated, False if remote state was outdated
        """
        # Conflict resolution: use newer version or newer timestamp for same version
        should_update = (
            remote_state.version > self._local_state.version or
            (remote_state.version == self._local_state.version and
             remote_state.timestamp > self._local_state.timestamp)
        )

        if should_update:
            # Merge account states (remote takes precedence)
            for account_index, account_data in remote_state.account_states.items():
                self._local_state.account_states[account_index] = account_data

            # Merge pool metrics (sum them)
            self._local_state.total_requests += remote_state.total_requests
            self._local_state.total_failures += remote_state.total_failures

            # Update metadata
            self._local_state.version = max(self._local_state.version, remote_state.version) + 1
            self._local_state.last_sync_time = time.time()
            self._local_state.timestamp = time.time()

            logger.info(f"DIST: Merged remote state (version {remote_state.version})")
            return True
        else:
            logger.debug(f"DIST: Ignored outdated remote state (version {remote_state.version})")
            return False

    def save_state_snapshot(self) -> Optional[str]:
        """Save current state as a snapshot file.

        Returns:
            Path to snapshot file or None if failed
        """
        if not self.state_file:
            return None

        try:
            # Create snapshot filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_file = f"{self.state_file}.{timestamp}.snapshot"

            # Save snapshot
            state_dict = self._serialize_state(self._local_state)
            path = Path(snapshot_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

            # Track snapshot and manage retention
            self._backup_files.append(snapshot_file)
            self._manage_snapshots()

            logger.info(f"DIST: Saved state snapshot to {snapshot_file}")
            return snapshot_file
        except Exception as e:
            logger.error(f"DIST: Failed to save snapshot: {e}")
            return None

    def _manage_snapshots(self) -> None:
        """Manage snapshot retention (keep last 10 snapshots)."""
        if len(self._backup_files) > 10:
            old_snapshots = self._backup_files[:-10]
            self._backup_files = self._backup_files[-10:]

            # Delete old snapshots
            for snapshot in old_snapshots:
                try:
                    Path(snapshot).unlink()
                    logger.debug(f"DIST: Deleted old snapshot {snapshot}")
                except Exception as e:
                    logger.warning(f"DIST: Failed to delete snapshot {snapshot}: {e}")

    def save_local_state(self) -> bool:
        """Save current state to local file.

        Returns:
            True if successful, False otherwise
        """
        if not self.state_file:
            return False

        try:
            state_dict = self._serialize_state(self._local_state)
            path = Path(self.state_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")

            # Save backup periodically
            now = time.time()
            if now - self._last_backup_time >= self.backup_interval:
                self.save_state_snapshot()
                self._last_backup_time = now

            return True
        except Exception as e:
            logger.error(f"DIST: Failed to save state: {e}")
            return False

    async def periodic_sync(self) -> None:
        """Periodic state synchronization task."""
        while True:
            try:
                # Save local state
                self.save_local_state()

                # In a real implementation, this would sync with other nodes
                # For now, we'll just log the sync attempt
                if self._local_state.version > 0:
                    logger.debug(f"DIST: Periodic sync (version {self._local_state.version})")

                await asyncio.sleep(self.sync_interval)
            except asyncio.CancelledError:
                logger.info("DIST: Sync task cancelled")
                break
            except Exception as e:
                logger.error(f"DIST: Sync task error: {e}")
                await asyncio.sleep(self.sync_interval)

    def get_consistency_report(self) -> Dict[str, Any]:
        """Get consistency report for current state.

        Returns:
            Consistency report
        """
        now = time.time()
        age_seconds = now - self._local_state.timestamp if self._local_state.timestamp > 0 else 0

        return {
            "node_id": self.node_id,
            "version": self._local_state.version,
            "timestamp": self._local_state.timestamp,
            "age_seconds": age_seconds,
            "accounts_tracked": len(self._local_state.account_states),
            "total_requests": self._local_state.total_requests,
            "total_failures": self._local_state.total_failures,
            "failure_rate": (
                self._local_state.total_failures / max(1, self._local_state.total_requests)
                if self._local_state.total_requests > 0 else 0.0
            ),
            "last_sync_time": self._local_state.last_sync_time,
            "state_file": self.state_file,
            "snapshots_retained": len(self._backup_files),
        }

    def export_schema_info(self) -> Dict[str, Any]:
        """Export schema information for version compatibility.

        Returns:
            Schema information
        """
        return {
            "version": 1,
            "fields": {
                "account_states": "dict[int, dict]",
                "total_requests": "int",
                "total_failures": "int",
                "last_sync_time": "float",
                "version": "int",
                "node_id": "str",
                "timestamp": "float",
            },
            "compatibility": "backward_compatible",
            "migration_notes": "Added in version 1.0"
        }


# Backward compatibility aliases
DistributedBackend = DistributedPersistence