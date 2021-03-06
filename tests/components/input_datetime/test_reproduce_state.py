"""Test reproduce state for Input datetime."""
from homeassistant.core import State

from tests.common import async_mock_service


async def test_reproducing_states(hass, caplog):
    """Test reproducing Input datetime states."""
    hass.states.async_set(
        "input_datetime.entity_datetime",
        "2010-10-10 01:20:00",
        {"has_date": True, "has_time": True},
    )
    hass.states.async_set(
        "input_datetime.entity_time", "01:20:00", {"has_date": False, "has_time": True}
    )
    hass.states.async_set(
        "input_datetime.entity_date",
        "2010-10-10",
        {"has_date": True, "has_time": False},
    )

    datetime_calls = async_mock_service(hass, "input_datetime", "set_datetime")

    # These calls should do nothing as entities already in desired state
    await hass.helpers.state.async_reproduce_state(
        [
            State("input_datetime.entity_datetime", "2010-10-10 01:20:00"),
            State("input_datetime.entity_time", "01:20:00"),
            State("input_datetime.entity_date", "2010-10-10"),
        ],
        blocking=True,
    )

    assert len(datetime_calls) == 0

    # Test invalid state is handled
    await hass.helpers.state.async_reproduce_state(
        [State("input_datetime.entity_datetime", "not_supported")], blocking=True
    )

    assert "not_supported" in caplog.text
    assert len(datetime_calls) == 0

    # Make sure correct services are called
    await hass.helpers.state.async_reproduce_state(
        [
            State("input_datetime.entity_datetime", "2011-10-10 02:20:00"),
            State("input_datetime.entity_time", "02:20:00"),
            State("input_datetime.entity_date", "2011-10-10"),
            # Should not raise
            State("input_datetime.non_existing", "2010-10-10 01:20:00"),
        ],
        blocking=True,
    )

    valid_calls = [
        {
            "entity_id": "input_datetime.entity_datetime",
            "datetime": "2011-10-10 02:20:00",
        },
        {"entity_id": "input_datetime.entity_time", "time": "02:20:00"},
        {"entity_id": "input_datetime.entity_date", "date": "2011-10-10"},
    ]
    assert len(datetime_calls) == 3
    for call in datetime_calls:
        assert call.domain == "input_datetime"
        assert call.data in valid_calls
        valid_calls.remove(call.data)
