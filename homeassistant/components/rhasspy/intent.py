import logging
import asyncio

import pydash
from homeassistant.helpers import intent

from .const import (
    INTENT_IS_DEVICE_ON,
    INTENT_IS_DEVICE_OFF,
    INTENT_IS_COVER_OPEN,
    INTENT_IS_COVER_CLOSED,
    INTENT_DEVICE_STATE,
    INTENT_TRIGGER_AUTOMATION,
    INTENT_TRIGGER_AUTOMATION_LATER,
    INTENT_SET_TIMER,
    INTENT_TIMER_READY,
)

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class DeviceStateIntent(intent.IntentHandler):
    intent_type = INTENT_DEVICE_STATE
    slot_schema = {"name": str}

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)

        response = intent_obj.create_response()

        # Cheesy plural check
        verb = "are" if name.endswith("s") else "is"

        speech = f"{name} {verb} {state.state}."
        _LOGGER.info(speech)
        response.async_set_speech(speech)
        return response


def make_state_handler(intent_obj, states):
    class StateIntent(intent.IntentHandler):
        intent_type = intent_obj
        slot_schema = {"name": str}

        async def async_handle(self, intent_obj):
            hass = intent_obj.hass
            slots = self.async_validate_slots(intent_obj.slots)
            name = slots["name"]["value"]
            state = intent.async_match_state(hass, name)
            is_state = state.state.lower() in states

            response = intent_obj.create_response()

            confirm = "yes" if is_state else "no"
            verb = "are" if name.endswith("s") else "is"

            speech = f"{confirm}. {name} {verb} {state.state}."
            _LOGGER.info(speech)
            response.async_set_speech(speech)
            return response

    return StateIntent()


class TriggerAutomationIntent(intent.IntentHandler):
    intent_type = INTENT_TRIGGER_AUTOMATION
    slot_schema = {"name": str}

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)

        await hass.services.async_call(
            "automation", "trigger", {"entity_id": state.entity_id}
        )

        response = intent_obj.create_response()
        return response


class SetTimerIntent(intent.IntentHandler):
    intent_type = INTENT_SET_TIMER
    slot_schema = {"hours": str, "minutes": str, "seconds": str}

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        total_seconds = SetTimerIntent.get_seconds(slots)

        _LOGGER.info(f"Waiting for {total_seconds} second(s)")
        await asyncio.sleep(total_seconds)

        return await intent.async_handle(hass, DOMAIN, INTENT_TIMER_READY, {}, "")

    @classmethod
    def get_seconds(cls, slots) -> int:
        # Compute total number of seconds for timer.
        # Time unit values may have multiple parts, like "30 2" for 32.
        total_seconds = 0
        for seconds_str in pydash.get(slots, "seconds.value").strip().split():
            total_seconds += int(seconds_str)

        for minutes_str in pydash.get(slots, "minutes.value", "").strip().split():
            total_seconds += int(minutes_str) * 60

        for hours_str in pydash.get(slots, "hours.value", "").strip().split():
            total_seconds += int(hours_str) * 60 * 60

        return total_seconds


class TriggerAutomationLaterIntent(intent.IntentHandler):
    intent_type = INTENT_TRIGGER_AUTOMATION_LATER
    slot_schema = {"name": str, "hours": str, "minutes": str, "seconds": str}

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)
        total_seconds = SetTimerIntent.get_seconds(slots)

        _LOGGER.info(f"Waiting for {total_seconds} second(s) before triggering {name}")
        await asyncio.sleep(total_seconds)

        # Trigger automation
        await hass.services.async_call(
            "automation", "trigger", {"entity_id": state.entity_id}
        )

        response = intent_obj.create_response()
        return response
