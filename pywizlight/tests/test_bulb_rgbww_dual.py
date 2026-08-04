"""Tests for the Bulb API."""

from typing import AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, call, patch

import pytest

from pywizlight import wizlight
from pywizlight.bulb import PilotParser
from pywizlight.bulblibrary import BulbClass, BulbType, Features, KelvinRange
from pywizlight.exceptions import WizLightConnectionError
from pywizlight.tests.fake_bulb import startup_bulb


@pytest.fixture()
async def rgbww_bulb() -> AsyncGenerator[wizlight, None]:
    shutdown, port = await startup_bulb(
        module_name="ESP20_DHRGB_01", firmware_version="1.35.0"
    )
    bulb = wizlight(ip="127.0.0.1", port=port)
    yield bulb
    await bulb.async_close()
    shutdown()


@pytest.mark.asyncio
async def test_model_description_rgbww_bulb(rgbww_bulb: wizlight) -> None:
    """Test fetching the model description rgbww bulb."""
    bulb_type = await rgbww_bulb.get_bulbtype()
    assert bulb_type == BulbType(
        features=Features(
            color=True, color_tmp=True, effect=True, brightness=True, dual_head=True
        ),
        name="ESP20_DHRGB_01",
        kelvin_range=KelvinRange(max=6500, min=2200),
        bulb_type=BulbClass.RGB,
        fw_version="1.35.0",
        white_channels=2,
        white_to_color_ratio=20,
    )


@pytest.mark.asyncio
async def test_dual_head_state_is_seeded_by_plain_get_pilot(
    rgbww_bulb: wizlight,
) -> None:
    """Test a plain getPilot seeds zoned state until pushes arrive."""
    states = await rgbww_bulb.updateState()

    assert states is not None
    assert len(states) == 2
    assert states[0] is not None
    assert states[1] is None


@pytest.mark.asyncio
async def test_dual_head_state_uses_strict_indexed_responses(
    rgbww_bulb: wizlight,
) -> None:
    """Test indexed responses map their 1-based tags to state slots."""
    await rgbww_bulb.get_bulbtype()
    plain = {"state": True, "dimming": 50}
    zone_a = {"devices": 1, "state": True, "dimming": 25}
    zone_b = {"devices": 2, "state": False, "dimming": 75}
    send = AsyncMock(
        side_effect=[
            {"method": "getPilot", "result": plain},
            {"method": "getPilot", "result": zone_a},
            {"method": "getPilot", "result": zone_b},
        ]
    )

    with patch.object(rgbww_bulb, "send", send):
        states = await rgbww_bulb.updateState()

    assert states is not None
    assert [state.pilotResult if state else None for state in states] == [
        zone_a,
        zone_b,
    ]
    assert send.call_args_list == [
        call({"method": "getPilot", "params": {}}),
        call({"method": "getPilot", "params": {"devices": 0}}),
        call({"method": "getPilot", "params": {"devices": 1}}),
    ]


@pytest.mark.asyncio
async def test_dual_head_state_ignores_untagged_indexed_responses(
    rgbww_bulb: wizlight,
) -> None:
    """Test ignored selectors do not fabricate duplicate zone states."""
    await rgbww_bulb.get_bulbtype()
    plain = {"state": True, "dimming": 50}
    send = AsyncMock(
        side_effect=[
            {"method": "getPilot", "result": plain},
            {"method": "getPilot", "result": plain},
            {"method": "getPilot", "result": {**plain, "devices": 1}},
        ]
    )

    with patch.object(rgbww_bulb, "send", send):
        states = await rgbww_bulb.updateState()

    assert states is not None
    assert states[0] is not None
    assert states[0].pilotResult == plain
    assert states[1] is None


@pytest.mark.asyncio
async def test_dual_head_state_keeps_partial_indexed_result_on_error(
    rgbww_bulb: wizlight,
) -> None:
    """Test an accepted head survives a later indexed request failure."""
    await rgbww_bulb.get_bulbtype()
    plain = {"state": True, "dimming": 50}
    zone_a = {"devices": 1, "state": True, "dimming": 25}
    send = AsyncMock(
        side_effect=[
            {"method": "getPilot", "result": plain},
            {"method": "getPilot", "result": zone_a},
            WizLightConnectionError("Invalid params"),
        ]
    )

    with patch.object(rgbww_bulb, "send", send):
        states = await rgbww_bulb.updateState()

    assert states is not None
    assert states[0] is not None
    assert states[0].pilotResult == zone_a
    assert states[1] is None


@pytest.mark.asyncio
async def test_dual_head_state_retries_indexed_polling_after_error(
    rgbww_bulb: wizlight,
) -> None:
    """Test an indexed request failure is retried on the next stale poll."""
    await rgbww_bulb.get_bulbtype()
    plain = {"state": True, "dimming": 50}
    send = AsyncMock(
        side_effect=[
            {"method": "getPilot", "result": plain},
            WizLightConnectionError("Invalid params"),
            {"method": "getPilot", "result": plain},
            WizLightConnectionError("Invalid params"),
        ]
    )

    with patch.object(rgbww_bulb, "send", send):
        await rgbww_bulb.updateState()
        await rgbww_bulb.updateState()

    assert send.call_args_list == [
        call({"method": "getPilot", "params": {}}),
        call({"method": "getPilot", "params": {"devices": 0}}),
        call({"method": "getPilot", "params": {}}),
        call({"method": "getPilot", "params": {"devices": 0}}),
    ]


@pytest.mark.asyncio
async def test_dual_head_pushes_update_individual_states(
    rgbww_bulb: wizlight,
) -> None:
    """Test indexed syncPilot pushes update only their corresponding head."""
    await rgbww_bulb.updateState()
    callbacks: List[List[Optional[PilotParser]]] = []
    rgbww_bulb.push_callback = lambda states: callbacks.append(states.copy())
    zone_a = {"devices": 1, "state": True, "dimming": 25}
    zone_b = {"devices": 2, "state": False, "dimming": 75}

    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_a}, ("127.0.0.1", 38899))
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_b}, ("127.0.0.1", 38899))

    assert rgbww_bulb.state[0] is not None
    assert rgbww_bulb.state[0].pilotResult == zone_a
    assert rgbww_bulb.state[1] is not None
    assert rgbww_bulb.state[1].pilotResult == zone_b
    assert len(callbacks) == 2

    rgbww_bulb.last_push = 0
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_b}, ("127.0.0.1", 38899))
    assert len(callbacks) == 2
    assert rgbww_bulb.last_push > 0


@pytest.mark.asyncio
async def test_dual_head_push_without_index_does_not_replace_zoned_state(
    rgbww_bulb: wizlight,
) -> None:
    """Test an ambiguous push does not collapse an established zoned state."""
    await rgbww_bulb.updateState()
    zone_a = {"devices": 1, "state": True}
    zone_b = {"devices": 2, "state": False}
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_a}, ("127.0.0.1", 38899))
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_b}, ("127.0.0.1", 38899))

    rgbww_bulb._on_push(
        {"method": "syncPilot", "params": {"state": False}},
        ("127.0.0.1", 38899),
    )

    assert rgbww_bulb.state[0] is not None
    assert rgbww_bulb.state[0].pilotResult == zone_a
    assert rgbww_bulb.state[1] is not None
    assert rgbww_bulb.state[1].pilotResult == zone_b

    rgbww_bulb.last_push = 0
    rgbww_bulb._on_push(
        {"method": "syncPilot", "params": {"state": True}},
        ("127.0.0.1", 38899),
    )
    assert rgbww_bulb.last_push == 0


@pytest.mark.asyncio
async def test_dual_head_poll_preserves_push_state(rgbww_bulb: wizlight) -> None:
    """Test polling does not discard per-head state learned from pushes."""
    await rgbww_bulb.updateState()
    zone_a = {"devices": 1, "state": True}
    zone_b = {"devices": 2, "state": False}
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_a}, ("127.0.0.1", 38899))
    rgbww_bulb._on_push({"method": "syncPilot", "params": zone_b}, ("127.0.0.1", 38899))
    rgbww_bulb.last_push = 0

    states = await rgbww_bulb.updateState()

    assert states is not None
    assert states[0] is not None
    assert states[0].pilotResult == zone_a
    assert states[1] is not None
    assert states[1].pilotResult == zone_b


@pytest.mark.asyncio
async def test_dual_head_push_during_poll_wins(rgbww_bulb: wizlight) -> None:
    """Test a push received during polling is not overwritten."""
    await rgbww_bulb.updateState()
    rgbww_bulb.last_push = 0
    original_send = rgbww_bulb.send
    zone_b = {"devices": 2, "state": True}

    async def send_with_push(message: dict) -> dict:
        response = await original_send(message)
        if message["params"].get("devices") == 0:
            rgbww_bulb._on_push(
                {"method": "syncPilot", "params": zone_b},
                ("127.0.0.1", 38899),
            )
            raise WizLightConnectionError("Invalid params")
        return response

    with patch.object(rgbww_bulb, "send", side_effect=send_with_push):
        states = await rgbww_bulb.updateState()

    assert states is not None
    assert states[1] is not None
    assert states[1].pilotResult == zone_b
