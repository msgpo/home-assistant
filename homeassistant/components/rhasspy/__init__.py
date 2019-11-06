"""Support for Rhasspy integration."""
import io
import logging
import threading
import time
import asyncio
from urllib.parse import urljoin

import pydash
import requests
import voluptuous as vol

import homeassistant.helpers.config_validation as cv
from homeassistant.core import Event, callback
from homeassistant.const import EVENT_COMPONENT_LOADED
from homeassistant.helpers import intent
from homeassistant.components.conversation import async_set_agent
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER

from .const import DOMAIN, SUPPORT_LANGUAGES
from .conversation import RhasspyConversationAgent


# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# Base URL of Rhasspy web server
CONF_WEB_URL = "web_url"

# Language to use for generating default utterances
CONF_LANGUAGE = "language"

# User defined utterances by intent
CONF_INTENTS = "intents"

# User defined slots and values
CONF_SLOTS = "slots"

# User defined words and pronunciations

CONF_CUSTOM_WORDS = "custom_words"

# Automatically generate turn on/off and toggle utterances for all entities with
# friendly names.
CONF_GENERATE_UTTERANCES = "generate_utterances"

# If True, Rhasspy conversation agent is registered
CONF_REGISTER_CONVERSATION = "register_conversation"

# List of entity ids by intent to generate uttrances for
CONF_INTENT_ENTITIES = "intent_entities"

# List of Home Assistant domains by intent to generate utterances for
CONF_INTENT_DOMAINS = "intent_domains"

# List of state names by intent to consider as matches
CONF_INTENT_STATES = "intent_states"

# List of format strings by intent.
# Used to automatically generate utterances.
CONF_UTTERANCE_TEMPLATES = "utterance_templates"

# Default settings
DEFAULT_WEB_URL = "http://localhost:12101"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_GENERATE_UTTERANCES = True
DEFAULT_REGISTER_CONVERSATION = True

# Custom intents
INTENT_IS_DEVICE_ON = "IsDeviceOn"
INTENT_IS_DEVICE_OFF = "IsDeviceOff"

INTENT_IS_COVER_OPEN = "IsCoverOpen"
INTENT_IS_COVER_CLOSED = "IsCoverClosed"

INTENT_DEVICE_STATE = "DeviceState"

INTENT_TRIGGER_AUTOMATION = "TriggerAutomation"
INTENT_TRIGGER_AUTOMATION_LATER = "TriggerAutomationLater"

INTENT_SET_TIMER = "SetTimer"

INTENT_TIMER_READY = "TimerReady"

# TODO: Translate to all supported languages
DEFAULT_UTTERANCES = {
    "en-US": {
        # Built-in intents
        intent.INTENT_TURN_ON: [
            "turn on [the|a|an] ({name}){{name}}",
            "turn [the|a|an] ({name}){{name}} on",
        ],
        intent.INTENT_TURN_OFF: [
            "turn off [the|a|an] ({name}){{name}}",
            "turn [the|a|an] ({name}){{name}} off",
        ],
        intent.INTENT_TOGGLE: [
            "toggle [the|a|an] ({name}){{name}}",
            "[the|a|an] ({name}){{name}} toggle",
        ],
        INTENT_OPEN_COVER: [
            "open [the|a|an] ({name}){{name}}",
            "[the|a|an] ({name}){{name}} open",
        ],
        INTENT_CLOSE_COVER: [
            "close [the|a|an] ({name}){{name}}",
            "[the|a|an] ({name}){{name}} close",
        ],
        # Custom intents
        INTENT_IS_DEVICE_ON: ["(is | are) [the|a|an] ({name}){{name}} on"],
        INTENT_IS_DEVICE_OFF: ["(is | are) [the|a|an] ({name}){{name}} off"],
        INTENT_DEVICE_STATE: [
            "what (is | are) [the|a|an] (state | states) of [the|a|an] ({name}){{name}}",
            "what [is | are] [the|a|an] ({name}){{name}} (state | states)",
        ],
        INTENT_IS_COVER_OPEN: ["(is | are) [the|a|an] ({name}){{name}} open"],
        INTENT_IS_COVER_CLOSED: ["(is | are) [the|a|an] ({name}){{name}} closed"],
        INTENT_TRIGGER_AUTOMATION: [
            "(run | trigger) [program | automation] ({name}){{name}}"
        ],
        INTENT_SET_TIMER: [
            "two_to_nine = (two:2 | three:3 | four:4 | five:5 | six:6 | seven:7 | eight:8 | nine:9)",
            "one_to_nine = (one:1 | <two_to_nine>)",
            "teens = (ten:10 | eleven:11 | twelve:12 | thirteen:13 | fourteen:14 | fifteen:15 | sixteen:16 | seventeen:17 | eighteen:18 | nineteen:19)",
            "tens = (twenty:20 | thirty:30 | forty:40 | fifty:50)",
            "one_to_nine = (one:1 | <two_to_nine>)",
            "one_to_fifty_nine = (<one_to_nine> | <teens> | <tens> [<one_to_nine>])",
            "two_to_fifty_nine = (<two_to_nine> | <teens> | <tens> [<one_to_nine>])",
            "hour_half_expr = (<one_to_nine>{{hours}} and (a half){{minutes:30}})",
            "hour_expr = (((one:1){{hours}}) | ((<one_to_nine>){{hours}}) | <hour_half_expr>) (hour | hours)",
            "minute_half_expr = (<one_to_fifty_nine>{{minutes}} and (a half){{seconds:30}})",
            "minute_expr = (((one:1){{minutes}}) | ((<two_to_fifty_nine>){{minutes}}) | <minute_half_expr>) (minute | minutes)",
            "second_expr = (((one:1){{seconds}}) | ((<two_to_fifty_nine>){{seconds}})) (second | seconds)",
            "time_expr = ((<hour_expr> [[and] <minute_expr>] [[and] <second_expr>]) | (<minute_expr> [[and] <second_expr>]) | <second_expr>)",
            "set [a] timer for <time_expr>",
        ],
        INTENT_TRIGGER_AUTOMATION_LATER: [
            "(run | trigger) [program | automation] ({name}){{name}} (in | after) <SetTimer.time_expr>",
            "(in | after) <SetTimer.time_expr> (run | trigger) [program | automation] ({name}){{name}}",
        ],
    }
}

DOMAIN_HOME_ASSISTANT = "homeassistant"

DEFAULT_INTENT_ENTITIES = {
    intent.INTENT_TURN_ON: ["group.all_lights", "group.all_switches"],
    intent.INTENT_TURN_OFF: ["group.all_lights", "group.all_switches"],
    intent.INTENT_TOGGLE: ["group.all_lights", "group.all_switches"],
    INTENT_IS_DEVICE_ON: ["group.all_lights", "group.all_switches"],
    INTENT_IS_DEVICE_OFF: ["group.all_lights", "group.all_switches"],
    INTENT_DEVICE_STATE: ["group.all_lights", "group.all_switches", "group.all_covers"],
    INTENT_OPEN_COVER: ["group.all_covers"],
    INTENT_CLOSE_COVER: ["group.all_covers"],
    INTENT_IS_COVER_OPEN: ["group.all_covers"],
    INTENT_IS_COVER_CLOSED: ["group.all_covers"],
    INTENT_SET_TIMER: [],
}

DEFAULT_INTENT_DOMAINS = {
    intent.INTENT_TURN_ON: ["light", "switch"],
    intent.INTENT_TURN_OFF: ["light", "switch"],
    intent.INTENT_TOGGLE: ["light", "switch"],
    INTENT_IS_DEVICE_ON: ["light", "switch", "binary_sensor"],
    INTENT_IS_DEVICE_OFF: ["light", "switch", "binary_sensor"],
    INTENT_DEVICE_STATE: ["light", "switch", "binary_sensor", "sensor", "cover"],
    INTENT_OPEN_COVER: ["cover"],
    INTENT_CLOSE_COVER: ["cover"],
    INTENT_IS_COVER_OPEN: ["cover"],
    INTENT_IS_COVER_CLOSED: ["cover"],
    INTENT_TRIGGER_AUTOMATION: ["automation"],
    INTENT_TRIGGER_AUTOMATION_LATER: ["automation"],
    INTENT_SET_TIMER: [DOMAIN_HOME_ASSISTANT],
}

DEFAULT_INTENT_STATES = {
    INTENT_IS_DEVICE_ON: ["on"],
    INTENT_IS_DEVICE_OFF: ["off"],
    INTENT_IS_COVER_OPEN: ["open"],
    INTENT_IS_COVER_CLOSED: ["closed"],
}

# Config
CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.All(
            {
                vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): vol.All(
                    cv.string, vol.In(SUPPORT_LANGUAGES)
                ),
                vol.Optional(CONF_WEB_URL, default=DEFAULT_WEB_URL): cv.url,
                vol.Optional(
                    CONF_GENERATE_UTTERANCES, default=DEFAULT_GENERATE_UTTERANCES
                ): bool,
                vol.Optional(
                    CONF_REGISTER_CONVERSATION, default=DEFAULT_REGISTER_CONVERSATION
                ): bool,
                vol.Optional(CONF_INTENTS): dict,
                vol.Optional(CONF_SLOTS): dict,
                vol.Optional(CONF_CUSTOM_WORDS): dict,
                vol.Optional(
                    CONF_INTENT_ENTITIES, default=DEFAULT_INTENT_ENTITIES
                ): dict,
                vol.Optional(CONF_INTENT_DOMAINS, default=DEFAULT_INTENT_DOMAINS): dict,
                vol.Optional(CONF_INTENT_STATES, default=DEFAULT_INTENT_STATES): dict,
                vol.Optional(CONF_UTTERANCE_TEMPLATES): dict,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)

# Services
SERVICE_TRAIN = "train"

SCHEMA_SERVICE_TRAIN = vol.Schema({})

# -----------------------------------------------------------------------------


async def async_setup(hass, config):
    """Set up Rhasspy integration."""

    conf = config.get(DOMAIN)
    web_url = conf.get(CONF_WEB_URL, DEFAULT_WEB_URL)

    register_conversation = conf.get(
        CONF_REGISTER_CONVERSATION, DEFAULT_REGISTER_CONVERSATION
    )

    if register_conversation:
        # Register converation agent
        agent = RhasspyConversationAgent(hass, web_url)
        async_set_agent(hass, agent)

        _LOGGER.info("Registered Rhasspy conversation agent")

    generate_utterances = conf.get(
        CONF_GENERATE_UTTERANCES, DEFAULT_GENERATE_UTTERANCES
    )
    if generate_utterances:
        # Register intent handlers
        intent.async_register(hass, DeviceStateIntent())
        intent.async_register(hass, TriggerAutomationIntent())
        intent.async_register(hass, TriggerAutomationLaterIntent())
        intent.async_register(hass, SetTimerIntent())

        intent_states = conf.get(CONF_INTENT_STATES, DEFAULT_INTENT_STATES)
        for intent_obj, states in intent_states.items():
            intent.async_register(hass, make_state_handler(intent_obj, states))

    provider = RhasspyProvider(hass, conf)
    await provider.async_initialize()

    hass.data[DOMAIN] = provider

    # Register services
    async def async_train_handle(service):
        """Service handle for train."""
        _LOGGER.info("Re-training profile")
        provider.schedule_retrain()

    hass.services.async_register(
        DOMAIN, SERVICE_TRAIN, async_train_handle, schema=SCHEMA_SERVICE_TRAIN
    )

    return True


# -----------------------------------------------------------------------------


class RhasspyProvider:
    def __init__(self, hass, config):
        self.hass = hass
        self.config = config

        # Base URL of Rhasspy web server
        self.url = config.get(CONF_WEB_URL, DEFAULT_WEB_URL)

        # URL to POST sentences.ini
        self.sentences_url = urljoin(self.url, "api/sentences")

        # URL to POST custom_words.txt
        self.custom_words_url = urljoin(self.url, "api/custom-words")

        # URL to POST slots
        self.slots_url = urljoin(self.url, "api/slots")

        # URL to train profile
        self.train_url = urljoin(self.url, "api/train")

        # e.g., en-US
        self.language = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        # entity id -> friendly name
        self.entities = {}

        self.generate_utterances = self.config.get(
            CONF_GENERATE_UTTERANCES, DEFAULT_GENERATE_UTTERANCES
        )

        # Threads/events
        self.train_thread = None
        self.train_event = threading.Event()

        self.train_timer_thread = None
        self.train_timer_event = threading.Event()
        self.train_timer_seconds = 1

    # -------------------------------------------------------------------------

    async def async_initialize(self):
        """Initialize Rhasspy provider."""

        # Register for component loaded event
        self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, self.component_loaded)

    @callback
    def component_loaded(self, event):
        """Handle a new component loaded."""
        old_entity_count = len(self.entities)
        for state in self.hass.states.async_all():
            if "_" not in state.name:
                self.entities[state.entity_id] = state

        if len(self.entities) > old_entity_count:
            _LOGGER.info("Need to retrain profile")
            self.schedule_retrain()

    # -------------------------------------------------------------------------

    def schedule_retrain(self):
        """Resets re-train timer."""
        if self.train_thread is None:
            self.train_thread = threading.Thread(
                target=self._training_thread_proc, daemon=True
            )
            self.train_thread.start()

        if self.train_timer_thread is None:
            self.train_timer_thread = threading.Thread(
                target=self._training_timer_thread_proc, daemon=True
            )
            self.train_timer_thread.start()

        # Reset timer
        self.train_timer_seconds = 1
        self.train_timer_event.set()

    def _training_thread_proc(self):
        """Re-trains voice2json provider. Works with a timer to avoid too many re-trains."""
        while True:
            # Wait for re-train request
            self.train_event.wait()
            self.train_event.clear()

            try:
                # rhasspy.intents
                config_utterances = self.config.get("intents", {})

                # Get utterance templates in the following order:
                # 1. rhasspy.utterance_templates
                # 2. Default templates for language
                # 3. Default English templates
                utterance_templates = self.config.get(
                    CONF_UTTERANCE_TEMPLATES,
                    DEFAULT_UTTERANCES.get(
                        self.language, DEFAULT_UTTERANCES[DEFAULT_LANGUAGE]
                    ),
                )

                # rhasspy.intent_domains
                intent_domains = self.config.get(
                    "intent_domains", DEFAULT_INTENT_DOMAINS
                )

                # rhasspy.intent_entities
                intent_entities = self.config.get(
                    "intent_entities", DEFAULT_INTENT_ENTITIES
                )

                # rhasspy.intent_states
                intent_states = self.config.get("intent_states", DEFAULT_INTENT_STATES)

                # Generate turn on/off, etc. for valid entities
                if self.generate_utterances:
                    default_utterances = DEFAULT_UTTERANCES.get(self.language)
                    for intent_obj, intent_utterances in utterance_templates.items():
                        intent_domain = intent_domains[intent_obj] or []
                        current_utterances = []

                        # Generate utterance for each entity
                        for entity_id, entity_state in self.entities.items():
                            # Check if entity has been explicitly included
                            valid_ids = intent_entities.get(intent_obj, []) or []

                            if entity_id not in valid_ids:
                                # Check entity domain
                                if (intent_obj in intent_domains) and (
                                    entity_state.domain not in intent_domain
                                ):
                                    _LOGGER.debug(
                                        f"Excluding {entity_id} (domain: {entity_state.domain}) from intent {intent_obj}"
                                    )
                                    continue

                            # Generate utterances from format strings
                            for utt_format in intent_utterances:
                                current_utterances.append(
                                    utt_format.format(name=entity_state.name)
                                )

                        # Add once
                        if DOMAIN_HOME_ASSISTANT in intent_domain:
                            for utt_format in intent_utterances:
                                current_utterances.append(utt_format.format())

                        if len(current_utterances) > 0:
                            # Update config utterances
                            config_utterances[intent_obj] = current_utterances

                num_utterances = sum(len(u) for u in config_utterances.values())
                if num_utterances > 0:
                    _LOGGER.debug("Writing sentences ({self.sentences_url})")

                    # Generate custom sentences.ini
                    with io.StringIO() as sentences_file:
                        for intent_type, utterances in config_utterances.items():
                            print(f"[{intent_type}]", file=sentences_file)

                            for utterance in utterances:
                                if utterance.startswith("["):
                                    # Escape "[" at start
                                    utterance = f"\\{utterance}"

                                print(utterance, file=sentences_file)

                            print("", file=sentences_file)

                        # POST sentences.ini
                        requests.post(self.sentences_url, sentences_file.getvalue())

                # Check for custom words
                custom_words = self.config.get("custom_words", {})
                if len(custom_words) > 0:
                    _LOGGER.debug(f"Writing custom words ({self.custom_words_url})")

                    with io.StringIO() as custom_words_file:
                        for word, pronunciations in custom_words.items():
                            # Accept either string or list of strings
                            if isinstance(pronunciations, str):
                                pronunciations = [pronunciations]

                            # word P1 P2 P3...
                            for pronunciation in pronunciations:
                                print(
                                    word.strip(),
                                    pronunciation.strip(),
                                    file=custom_words_file,
                                )

                        # POST custom_words.txt
                        requests.post(
                            self.custom_words_url, custom_words_file.getvalue()
                        )

                # Check for slots
                slots = self.config.get("slots", {})
                if len(slots) > 0:
                    _LOGGER.debug(f"Writing slots ({self.slots_url})")
                    for slot_name, slot_values in list(slots.items()):
                        # Accept either string or list of strings
                        if isinstance(slot_values, str):
                            slots[slot_name] = [slot_values]

                    # POST slots (JSON)
                    requests.post(self.slots_url, json=slots)

                # Train profile
                _LOGGER.info(f"Training profile ({self.train_url})")
                requests.post(self.train_url)

                _LOGGER.info("Ready")
            except Exception as e:
                _LOGGER.exception("train")

    def _training_timer_thread_proc(self):
        """Counts down a timer and triggers a re-train when it reaches zero."""
        while True:
            self.train_timer_event.wait()

            while self.train_timer_seconds > 0:
                time.sleep(0.1)
                self.train_timer_seconds -= 0.1

            self.train_event.set()
            self.train_timer_event.clear()


# -----------------------------------------------------------------------------


class DeviceStateIntent(intent.IntentHandler):
    intent_type = INTENT_DEVICE_STATE
    slot_schema = {"name": cv.string}

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
        slot_schema = {"name": cv.string}

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
    slot_schema = {"name": cv.string}

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
    slot_schema = {"hours": cv.string, "minutes": cv.string, "seconds": cv.string}

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
    slot_schema = {
        "name": cv.string,
        "hours": cv.string,
        "minutes": cv.string,
        "seconds": cv.string,
    }

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
