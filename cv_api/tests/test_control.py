"""Tests for validated occupancy control commands."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analytics.schemas import OccupancyControlCommand


def test_control_command_accepts_supported_action_and_operator() -> None:
    command = OccupancyControlCommand(
        camera_id=" cam1 ",
        action="confirm_present",
        operator_id=" operator-7 ",
    )
    assert command.camera_id == "cam1"
    assert command.operator_id == "operator-7"


@pytest.mark.parametrize(
    "payload",
    [
        {"camera_id": "cam1", "action": "reset"},
        {"camera_id": "cam1", "action": "confirm_present", "unexpected": True},
        {"camera_id": "cam1"},
    ],
)
def test_control_command_rejects_unsupported_or_unknown_fields(payload: dict) -> None:
    with pytest.raises(ValidationError):
        OccupancyControlCommand.model_validate(payload)
