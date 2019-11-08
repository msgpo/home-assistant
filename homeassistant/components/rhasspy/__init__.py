"""Support for Rhasspy integration."""
import io
import logging
import threading
import time
import asyncio
from collections import defaultdict
from urllib.parse import urljoin

import pydash
import requests
import voluptuous as vol

import homeassistant.helpers.config_validation as cv
from homeassistant.core import Event, callback, State
from homeassistant.const import EVENT_COMPONENT_LOADED
from homeassistant.helpers import intent
from homeassistant.helpers.template import Template as T
from homeassistant.components.conversation import async_set_agent
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER

from .const import (
    DOMAIN,
    SUPPORT_LANGUAGES,
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

from .conversation import RhasspyConversationAgent
from .core import command_to_sentences


# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# Base URL of Rhasspy web server
CONF_WEB_URL = "web_url"

# Language to use for generating default utterances
CONF_LANGUAGE = "language"

# User defined commands by intent
CONF_COMMANDS = "commands"

# User defined slots and values
CONF_SLOTS = "slots"

# User defined words and pronunciations
CONF_CUSTOM_WORDS = "custom_words"

# Automatically generate turn on/off and toggle utterances for all entities with
# friendly names.
# CONF_GENERATE_UTTERANCES = "generate_utterances"

# If True, Rhasspy conversation agent is registered
CONF_REGISTER_CONVERSATION = "register_conversation"

# List of entity ids by intent to generate uttrances for
# CONF_INTENT_ENTITIES = "intent_entities"

# List of Home Assistant domains by intent to generate utterances for
# CONF_INTENT_DOMAINS = "intent_domains"

# List of state names by intent to consider as matches
# CONF_INTENT_STATES = "intent_states"

# List of format strings by intent.
# Used to automatically generate utterances.
# CONF_INTENT_TEMPLATES = "intent_templates"

# Default settings
DEFAULT_WEB_URL = "http://localhost:12101"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_COMMANDS = {}
DEFAULT_SLOTS = {}
DEFAULT_CUSTOM_WORDS = {}
DEFAULT_REGISTER_CONVERSATION = True

# KEY_INCLUDE_DOMAINS = "include_domains"
# KEY_INCLUDE_ENTITIES = "include_entities"
# KEY_EXCLUDE_DOMAINS = "exclude_domains"
# KEY_EXCLUDE_ENTITIES = "exclude_entities"

# DEFAULT_GENERATE_UTTERANCES = {
#     intent.INTENT_TURN_ON: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch"],
#         KEY_INCLUDE_ENTITIES: ["group.all_lights", "group.all_switches"],
#     },
#     intent.INTENT_TURN_OFF: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch"],
#         KEY_INCLUDE_ENTITIES: ["group.all_lights", "group.all_switches"],
#     },
#     intent.INTENT_TOGGLE: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch"],
#         KEY_INCLUDE_ENTITIES: ["group.all_lights", "group.all_switches"],
#     },
#     INTENT_OPEN_COVER: {
#         KEY_INCLUDE_DOMAINS: ["cover"],
#         KEY_INCLUDE_ENTITIES: ["group.all_covers"],
#     },
#     INTENT_CLOSE_COVER: {
#         KEY_INCLUDE_DOMAINS: ["cover"],
#         KEY_INCLUDE_ENTITIES: ["group.all_covers"],
#     },
#     INTENT_IS_DEVICE_ON: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch", "binary_sensor"],
#         KEY_INCLUDE_ENTITIES: ["group.all_lights", "group.all_switches"],
#     },
#     INTENT_IS_DEVICE_OFF: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch", "binary_sensor"],
#         KEY_INCLUDE_ENTITIES: ["group.all_lights", "group.all_switches"],
#     },
#     INTENT_IS_COVER_OPEN: {
#         KEY_INCLUDE_DOMAINS: ["cover"],
#         KEY_INCLUDE_ENTITIES: ["group.all_covers"],
#     },
#     INTENT_IS_COVER_CLOSED: {
#         KEY_INCLUDE_DOMAINS: ["cover"],
#         KEY_INCLUDE_ENTITIES: ["group.all_covers"],
#     },
#     INTENT_DEVICE_STATE: {
#         KEY_INCLUDE_DOMAINS: ["light", "switch", "binary_sensor", "sensor", "cover"],
#         KEY_INCLUDE_ENTITIES: [
#             "group.all_lights",
#             "group.all_switches",
#             "group.all_covers",
#         ],
#     },
#     INTENT_TRIGGER_AUTOMATION: {KEY_INCLUDE_DOMAINS: ["automation"]},
#     INTENT_TRIGGER_AUTOMATION_LATER: {KEY_INCLUDE_DOMAINS: ["automation"]},
# }

# TODO: Translate to all supported languages
DEFAULT_UTTERANCES = {
    "en-US": {
        # Built-in intents
        intent.INTENT_TURN_ON: [
            T("turn on [the|a|an] ({{ entity.name }}){name}"),
            T("turn [the|a|an] ({{ entity.name }}){name} on"),
        ],
        intent.INTENT_TURN_OFF: [
            T("turn off [the|a|an] ({{ entity.name }}){name}"),
            T("turn [the|a|an] ({{ entity.name }}){name} off"),
        ],
        intent.INTENT_TOGGLE: [
            T("toggle [the|a|an] ({{ entity.name }}){name}"),
            T("[the|a|an] ({{ entity.name }}){name} toggle"),
        ],
        INTENT_OPEN_COVER: [
            T("open [the|a|an] ({{ entity.name }}){name}"),
            T("[the|a|an] ({{ entity.name }}){name} open"),
        ],
        INTENT_CLOSE_COVER: [
            T("close [the|a|an] ({{ entity.name }}){name}"),
            T("[the|a|an] ({{ entity.name }}){name} close"),
        ],
        # Custom intents
        INTENT_IS_DEVICE_ON: [T("(is | are) [the|a|an] ({{ entity.name }}){name} on")],
        INTENT_IS_DEVICE_OFF: [
            T("(is | are) [the|a|an] ({{ entity.name }}){name} off")
        ],
        INTENT_DEVICE_STATE: [
            T(
                "what (is | are) [the|a|an] (state | states) of [the|a|an] ({{ entity.name }}){name}"
            ),
            T("what [is | are] [the|a|an] ({{ entity.name }}){name} (state | states)"),
        ],
        INTENT_IS_COVER_OPEN: [
            T("(is | are) [the|a|an] ({{ entity.name }}){name} open")
        ],
        INTENT_IS_COVER_CLOSED: [
            T("(is | are) [the|a|an] ({{ entity.name }}){name} closed")
        ],
        INTENT_TRIGGER_AUTOMATION: [
            T("(run | trigger) [program | automation] ({{ entity.name }}){name}")
        ],
        INTENT_SET_TIMER: [
            T(
                "two_to_nine = (two:2 | three:3 | four:4 | five:5 | six:6 | seven:7 | eight:8 | nine:9)"
            ),
            T("one_to_nine = (one:1 | <two_to_nine>)"),
            T(
                "teens = (ten:10 | eleven:11 | twelve:12 | thirteen:13 | fourteen:14 | fifteen:15 | sixteen:16 | seventeen:17 | eighteen:18 | nineteen:19)"
            ),
            T("tens = (twenty:20 | thirty:30 | forty:40 | fifty:50)"),
            T("one_to_nine = (one:1 | <two_to_nine>)"),
            T("one_to_fifty_nine = (<one_to_nine> | <teens> | <tens> [<one_to_nine>])"),
            T("two_to_fifty_nine = (<two_to_nine> | <teens> | <tens> [<one_to_nine>])"),
            T("hour_half_expr = (<one_to_nine>{hours} and (a half){{minutes:30}})"),
            T(
                "hour_expr = (((one:1){hours}) | ((<one_to_nine>){hours}) | <hour_half_expr>) (hour | hours)"
            ),
            T(
                "minute_half_expr = (<one_to_fifty_nine>{minutes} and (a half){{seconds:30}})"
            ),
            T(
                "minute_expr = (((one:1){minutes}) | ((<two_to_fifty_nine>){minutes}) | <minute_half_expr>) (minute | minutes)"
            ),
            T(
                "second_expr = (((one:1){seconds}) | ((<two_to_fifty_nine>){seconds})) (second | seconds)"
            ),
            T(
                "time_expr = ((<hour_expr> [[and] <minute_expr>] [[and] <second_expr>]) | (<minute_expr> [[and] <second_expr>]) | <second_expr>)"
            ),
            T("set [a] timer for <time_expr>"),
        ],
        INTENT_TRIGGER_AUTOMATION_LATER: [
            T(
                "(run | trigger) [program | automation] ({{ entity.name }}){name} (in | after) <SetTimer.time_expr>"
            ),
            T(
                "(in | after) <SetTimer.time_expr> (run | trigger) [program | automation] ({{ entity.name }}){name}"
            ),
        ],
    }
}

# DOMAIN_HOME_ASSISTANT = "homeassistant"

# DEFAULT_INTENT_ENTITIES = {
#     intent.INTENT_TURN_ON: ["group.all_lights", "group.all_switches"],
#     intent.INTENT_TURN_OFF: ["group.all_lights", "group.all_switches"],
#     intent.INTENT_TOGGLE: ["group.all_lights", "group.all_switches"],
#     INTENT_IS_DEVICE_ON: ["group.all_lights", "group.all_switches"],
#     INTENT_IS_DEVICE_OFF: ["group.all_lights", "group.all_switches"],
#     INTENT_DEVICE_STATE: ["group.all_lights", "group.all_switches", "group.all_covers"],
#     INTENT_OPEN_COVER: ["group.all_covers"],
#     INTENT_CLOSE_COVER: ["group.all_covers"],
#     INTENT_IS_COVER_OPEN: ["group.all_covers"],
#     INTENT_IS_COVER_CLOSED: ["group.all_covers"],
#     INTENT_SET_TIMER: [],
# }

# DEFAULT_INTENT_DOMAINS = {
#     intent.INTENT_TURN_ON: ["light", "switch"],
#     intent.INTENT_TURN_OFF: ["light", "switch"],
#     intent.INTENT_TOGGLE: ["light", "switch"],
#     INTENT_IS_DEVICE_ON: ["light", "switch", "binary_sensor"],
#     INTENT_IS_DEVICE_OFF: ["light", "switch", "binary_sensor"],
#     INTENT_DEVICE_STATE: ["light", "switch", "binary_sensor", "sensor", "cover"],
#     INTENT_OPEN_COVER: ["cover"],
#     INTENT_CLOSE_COVER: ["cover"],
#     INTENT_IS_COVER_OPEN: ["cover"],
#     INTENT_IS_COVER_CLOSED: ["cover"],
#     INTENT_TRIGGER_AUTOMATION: ["automation"],
#     INTENT_TRIGGER_AUTOMATION_LATER: ["automation"],
#     INTENT_SET_TIMER: [DOMAIN_HOME_ASSISTANT],
# }

# DEFAULT_INTENT_STATES = {
#     INTENT_IS_DEVICE_ON: ["on"],
#     INTENT_IS_DEVICE_OFF: ["off"],
#     INTENT_IS_COVER_OPEN: ["open"],
#     INTENT_IS_COVER_CLOSED: ["closed"],
# }

# DEFAULT_INTENT_RESPONSES: {INTENT_IS_DEVICE_ON: "{% %}"}

# Config
COMMAND_SCHEMA = vol.Schema(
    {
        vol.Exclusive("command", "commands"): str,
        vol.Exclusive("command_template", "commands"): cv.template,
        vol.Exclusive("commands", "commands"): vol.All(cv.ensure_list, [str]),
        vol.Exclusive("command_templates", "commands"): vol.All(
            cv.ensure_list, [cv.template]
        ),
        vol.Optional("data"): vol.Schema({str: object}),
        vol.Optional("data_template"): vol.Schema({str: cv.template}),
        vol.Optional("include"): vol.Schema(
            {vol.Optional("domains"): vol.All(cv.ensure_list, [str])},
            {vol.Optional("entities"): vol.All(cv.ensure_list, [cv.entity_id])},
        ),
        vol.Optional("exclude"): vol.Schema(
            {vol.Optional("entities"): vol.All(cv.ensure_list, [cv.entity_id])}
        ),
    }
)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.All(
            {
                vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): vol.All(
                    str, vol.In(SUPPORT_LANGUAGES)
                ),
                vol.Optional(CONF_WEB_URL, default=DEFAULT_WEB_URL): cv.url,
                vol.Optional(
                    CONF_REGISTER_CONVERSATION, default=DEFAULT_REGISTER_CONVERSATION
                ): bool,
                vol.Optional(CONF_SLOTS, DEFAULT_SLOTS): {
                    str: vol.All(cv.ensure_list, [str])
                },
                vol.Optional(CONF_CUSTOM_WORDS, DEFAULT_CUSTOM_WORDS): vol.Schema(
                    {str: str}
                ),
                vol.Optional(CONF_COMMANDS, DEFAULT_COMMANDS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str, COMMAND_SCHEMA])}
                ),
                # vol.Optional(
                #     CONF_INTENT_ENTITIES, default=DEFAULT_INTENT_ENTITIES
                # ): dict,
                # vol.Optional(CONF_INTENT_DOMAINS, default=DEFAULT_INTENT_DOMAINS): dict,
                # vol.Optional(CONF_INTENT_STATES, default=DEFAULT_INTENT_STATES): dict,
                # vol.Optional(CONF_UTTERANCE_TEMPLATES): vol.Schema(
                #     {str: vol.All(cv.ensure_list, [cv.template])}
                # ),
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
        self.url: str = config.get(CONF_WEB_URL, DEFAULT_WEB_URL)

        # URL to POST sentences.ini
        self.sentences_url: str = urljoin(self.url, "api/sentences")

        # URL to POST custom_words.txt
        self.custom_words_url: str = urljoin(self.url, "api/custom-words")

        # URL to POST slots
        self.slots_url: str = urljoin(self.url, "api/slots")

        # URL to train profile
        self.train_url: str = urljoin(self.url, "api/train")

        # e.g., en-US
        self.language: str = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        # entity id -> entity state
        self.entities: Dict[str, State] = {}

        # entity id -> name
        self.entity_name_map: Dict[str, str] = {}

        # Threads/events
        self.train_thread = None
        self.train_event = threading.Event()

        self.train_timer_thread = None
        self.train_timer_event = threading.Event()
        self.train_timer_seconds = 1

    # -------------------------------------------------------------------------

    async def async_initialize(self):
        """Initialize Rhasspy provider."""

        # Register intent handlers
        # for intent_obj in self.generate_utterances:
        #     if intent_obj == INTENT_DEVICE_STATE:
        #         intent.async_register(self.hass, DeviceStateIntent())
        #     elif intent_obj == INTENT_TRIGGER_AUTOMATION:
        #         intent.async_register(self.hass, TriggerAutomationIntent())
        #     elif intent_obj == INTENT_TRIGGER_AUTOMATION_LATER:
        #         intent.async_register(self.hass, TriggerAutomationLaterIntent())
        #     elif intent_obj == INTENT_SET_TIMER:
        #         intent.async_register(self.hass, SetTimerIntent())

        # intent_states = self.config.get(CONF_INTENT_STATES, DEFAULT_INTENT_STATES)
        # for intent_obj, states in intent_states.items():
        #     if intent_obj in self.generate_utterances:
        #         intent.async_register(self.hass, make_state_handler(intent_obj, states))

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
                sentences_by_intent: Dict[str, List[str]] = defaultdict(list)

                # Non-templated commands
                for intent_type, commands in self.config.get(
                    CONF_COMMANDS, DEFAULT_COMMANDS
                ).items():
                    for command in commands:
                        for sentence in command_to_sentences(
                            self.hass, command, self.entities
                        ):
                            sentences_by_intent[intent_type].append(sentence)

                # rhasspy.intents
                # config_utterances = self.config.get("intents", {})

                # Get utterance templates in the following order:
                # 1. rhasspy.utterance_templates
                # 2. Default templates for language
                # 3. Default English templates
                # utterance_templates = self.config.get(
                #     CONF_UTTERANCE_TEMPLATES,
                #     DEFAULT_UTTERANCES.get(
                #         self.language, DEFAULT_UTTERANCES[DEFAULT_LANGUAGE]
                #     ),
                # )

                # rhasspy.intent_domains
                # intent_domains = self.config.get(
                #     "intent_domains", DEFAULT_INTENT_DOMAINS
                # )

                # rhasspy.intent_entities
                # intent_entities = self.config.get(
                #     "intent_entities", DEFAULT_INTENT_ENTITIES
                # )

                # rhasspy.intent_states
                # intent_states = self.config.get("intent_states", DEFAULT_INTENT_STATES)

                # Generate turn on/off, etc. for valid entities
                # if len(self.generate_utterances) > 0:
                #     default_templates = DEFAULT_UTTERANCES.get(self.language)

                #     for intent_obj, intent_utterances in utterance_templates.items():
                #         if intent_obj not in self.generate_utterances:
                #             continue

                #         intent_domain = intent_domains.get(intent_obj) or []
                #         current_utterances = []

                #         # Generate utterance for each entity
                #         for entity_id, entity_state in self.entities.items():
                #             # Check if entity has been explicitly included
                #             valid_ids = intent_entities.get(intent_obj, []) or []

                #             if entity_id not in valid_ids:
                #                 # Check entity domain
                #                 if (intent_obj in intent_domains) and (
                #                     entity_state.domain not in intent_domain
                #                 ):
                #                     _LOGGER.debug(
                #                         f"Excluding {entity_id} (domain: {entity_state.domain}) from intent {intent_obj}"
                #                     )
                #                     continue

                #             # Generate utterances from format strings
                #             for utt_format in intent_utterances:
                #                 utt_format.hass = self.hass
                #                 current_utterances.append(
                #                     utt_format.async_render(state=entity_state)
                #                 )

                #         # Add once
                #         if DOMAIN_HOME_ASSISTANT in intent_domain:
                #             for utt_format in intent_utterances:
                #                 current_utterances.append(utt_format.format())

                #         if len(current_utterances) > 0:
                #             # Update config utterances
                #             config_utterances[intent_obj] = current_utterances

                #             _LOGGER.info(current_utterances)

                # Check for custom sentences
                num_sentences = sum(len(s) for s in sentences_by_intent.values())
                if num_sentences > 0:
                    _LOGGER.debug("Writing sentences ({self.sentences_url})")

                    # Generate custom sentences.ini
                    with io.StringIO() as sentences_file:
                        for intent_type, sentences in sentences_by_intent.items():
                            print(f"[{intent_type}]", file=sentences_file)

                            for sentence in sentences:
                                if sentence.startswith("["):
                                    # Escape "[" at start
                                    sentence = f"\\{utterance}"

                                print(sentence, file=sentences_file)

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
