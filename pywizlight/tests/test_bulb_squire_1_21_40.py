"""Tests for the Bulb API with a Squire."""

from typing import AsyncGenerator
from unittest.mock import call, patch

import pytest

from pywizlight import PilotBuilder, wizlight
from pywizlight.bulblibrary import BulbClass, BulbType, Features, KelvinRange
from pywizlight.tests.fake_bulb import startup_bulb


@pytest.fixture()
async def squire() -> AsyncGenerator[wizlight, None]:
    shutdown, port = await startup_bulb(
        module_name="ESP20_DHRGB_01B", firmware_version="1.21.40"
    )
    bulb = wizlight(ip="127.0.0.1", port=port)
    yield bulb
    await bulb.async_close()
    shutdown()


@pytest.mark.asyncio
async def test_setting_ratio(squire: wizlight) -> None:
    """Test setting ratio."""
    await squire.set_ratio(50)
    states = await squire.updateState()
    assert states is not None and len(states) == 1
    assert states and states[0] and states[0].get_ratio() == 50
    await squire.turn_on(PilotBuilder(ratio=20))
    states = await squire.updateState()
    assert states and states[0] and states[0].get_ratio() == 20
    with pytest.raises(ValueError):
        await squire.set_ratio(500)


@pytest.mark.asyncio
async def test_ratio_push_remains_single_state(squire: wizlight) -> None:
    """Test a ratio-based dual-head push updates one combined state."""
    await squire.get_bulbtype()
    await squire.set_ratio(50)
    with patch.object(squire, "send", wraps=squire.send) as send:
        await squire.updateState()

    assert send.call_args_list == [call({"method": "getPilot", "params": {}})]
    push_state = {"devices": 2, "state": True, "ratio": 75}

    squire._on_push({"method": "syncPilot", "params": push_state}, ("127.0.0.1", 38899))

    assert len(squire.state) == 1
    assert squire.state[0] is not None
    assert squire.state[0].pilotResult == push_state


@pytest.mark.asyncio
async def test_model_description_squire(squire: wizlight) -> None:
    """Test fetching the model description for a squire."""
    bulb_type = await squire.get_bulbtype()
    assert bulb_type == BulbType(
        features=Features(
            color=True, color_tmp=True, effect=True, brightness=True, dual_head=True
        ),
        name="ESP20_DHRGB_01B",
        kelvin_range=KelvinRange(max=6500, min=2200),
        bulb_type=BulbClass.RGB,
        fw_version="1.21.40",
        white_channels=2,
        white_to_color_ratio=20,
    )
