import logging
import asyncio
from typing import List

import pydash
from homeassistant.helpers import intent
from homeassistant.helpers.template import Template

from .const import (
    DOMAIN,
    INTENT_IS_DEVICE_ON,
    INTENT_IS_DEVICE_OFF,
    INTENT_IS_COVER_OPEN,
    INTENT_IS_COVER_CLOSED,
    INTENT_IS_DEVICE_STATE,
    INTENT_DEVICE_STATE,
    INTENT_TRIGGER_AUTOMATION,
    INTENT_TRIGGER_AUTOMATION_LATER,
    INTENT_SET_TIMER,
    INTENT_TIMER_READY,
)

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class IsDeviceStateIntent(intent.IntentHandler):
    """Confirms or disconfirms the variable state of a device."""

    intent_type = INTENT_IS_DEVICE_STATE
    slot_schema = {"name": str, "state": str}

    def __init__(self, speech_template: Template):
        self.speech_template = speech_template

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state_name = slots["state"]["value"]
        state = intent.async_match_state(hass, name)

        self.speech_template.hass = hass
        speech = self.speech_template.async_render(
            {"entity": state, "state": state_name}
        )
        _LOGGER.debug(speech)

        response = intent_obj.create_response()
        response.async_set_speech(speech)
        return response


class DeviceStateIntent(intent.IntentHandler):
    """Reports a device's state through speech."""

    intent_type = INTENT_DEVICE_STATE
    slot_schema = {"name": str}

    def __init__(self, speech_template: Template):
        self.speech_template = speech_template

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)

        self.speech_template.hass = hass
        speech = self.speech_template.async_render({"entity": state})
        _LOGGER.debug(speech)

        response = intent_obj.create_response()
        response.async_set_speech(speech)
        return response


# -----------------------------------------------------------------------------


def make_state_handler(intent_obj, states: List[str], speech_template: Template):
    class StateIntent(intent.IntentHandler):
        """Confirms or disconfirms the specific state of a device."""

        intent_type = intent_obj
        slot_schema = {"name": str}

        def __init__(self, states: List[str], speech_template: Template):
            self.speech_template = speech_template
            self.states = states

        async def async_handle(self, intent_obj):
            hass = intent_obj.hass
            slots = self.async_validate_slots(intent_obj.slots)
            name = slots["name"]["value"]
            state = intent.async_match_state(hass, name)

            self.speech_template.hass = hass
            speech = self.speech_template.async_render(
                {"entity": state, "states": self.states}
            )
            _LOGGER.debug(speech)

            response = intent_obj.create_response()
            response.async_set_speech(speech)
            return response

    return StateIntent(states, speech_template)


# -----------------------------------------------------------------------------


class SetTimerIntent(intent.IntentHandler):
    """Waits for a specified amount of time and then generates an INTENT_TIMER_READY."""

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


class TimerReadyIntent(intent.IntentHandler):
    """Generated after INTENT_SET_TIMER timeout elapses."""

    intent_type = INTENT_TIMER_READY

    def __init__(self, speech_template: Template):
        self.speech_template = speech_template

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        self.speech_template.hass = hass

        speech = self.speech_template.async_render()
        _LOGGER.debug(speech)

        response = intent_obj.create_response()
        response.async_set_speech(speech)
        return response


# -----------------------------------------------------------------------------


class TriggerAutomationIntent(intent.IntentHandler):
    """Triggers an automation by name and generates speech according to a template."""

    intent_type = INTENT_TRIGGER_AUTOMATION
    slot_schema = {"name": str}

    def __init__(self, speech_template: Template):
        self.speech_template = speech_template

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)

        await hass.services.async_call(
            "automation", "trigger", {"entity_id": state.entity_id}
        )

        self.speech_template.hass = hass
        speech = self.speech_template.async_render({"automation": state})
        _LOGGER.debug(speech)

        response = intent_obj.create_response()
        response.async_set_speech(speech)
        return response


class TriggerAutomationLaterIntent(intent.IntentHandler):
    """Waits for a specified amount of time and then triggers an automation using INTENT_TRIGGER_AUTOMATION."""

    intent_type = INTENT_TRIGGER_AUTOMATION_LATER
    slot_schema = {"name": str, "hours": str, "minutes": str, "seconds": str}

    async def async_handle(self, intent_obj):
        hass = intent_obj.hass
        slots = self.async_validate_slots(intent_obj.slots)
        name = slots["name"]["value"]
        state = intent.async_match_state(hass, name)
        total_seconds = SetTimerIntent.get_seconds(slots)

        _LOGGER.debug(f"Waiting for {total_seconds} second(s) before triggering {name}")
        await asyncio.sleep(total_seconds)

        # Trigger automation
        await hass.services.async_call(
            "automation", "trigger", {"entity_id": state.entity_id}
        )

        # Use INTENT_TRIGGER_AUTOMATION
        return await intent.async_handle(
            hass, DOMAIN, INTENT_TRIGGER_AUTOMATION, {"name": name}, ""
        )
