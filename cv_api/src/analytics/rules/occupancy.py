"""Occupancy state shared by line-crossing rules and operator controls."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock


@dataclass(frozen=True)
class OccupancyStatus:
    """Occupancy information for one camera."""

    count: int
    uncertain: bool
    uncertain_track_ids: tuple[int, ...]


class OccupancyManager:
    """Maintain confirmed and uncertain occupant track IDs."""

    def __init__(self) -> None:
        self._occupancy: dict[str, set[int]] = {}
        self._uncertain: dict[str, set[int]] = {}
        self._lock = RLock()

    def update(self, camera_id: str, track_id: int, movement: str) -> int:
        if movement not in {"entry", "exit"}:
            raise ValueError(f"Unsupported occupancy movement: {movement}")
        with self._lock:
            occupants = self._occupancy.setdefault(camera_id, set())
            uncertain = self._uncertain.setdefault(camera_id, set())
            if movement == "entry":
                occupants.add(track_id)
                uncertain.discard(track_id)
            else:
                occupants.discard(track_id)
                uncertain.discard(track_id)
            return len(occupants)

    def status(self, camera_id: str) -> OccupancyStatus:
        with self._lock:
            occupants = self._occupancy.get(camera_id, set())
            uncertain = self._uncertain.get(camera_id, set())
            return OccupancyStatus(len(occupants), bool(uncertain), tuple(sorted(uncertain)))

    def reconcile(self, camera_id: str, action: str) -> OccupancyStatus:
        if action not in {"confirm_present", "confirm_exited"}:
            raise ValueError(f"Unsupported occupancy reconciliation action: {action!r}")
        with self._lock:
            occupants = self._occupancy.setdefault(camera_id, set())
            uncertain = self._uncertain.setdefault(camera_id, set())
            if action == "confirm_exited":
                occupants.difference_update(uncertain)
            uncertain.clear()
        return self.status(camera_id)

    def mark_expired(self, camera_id: str, track_id: int) -> None:
        with self._lock:
            if track_id in self._occupancy.get(camera_id, set()):
                self._uncertain.setdefault(camera_id, set()).add(track_id)

    def reset(self) -> None:
        """Clear all occupancy state."""
        with self._lock:
            self._occupancy.clear()
            self._uncertain.clear()
