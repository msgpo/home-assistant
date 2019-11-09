"""Support for Rhasspy integration."""
import io
import re
import logging
import threading
import time
import asyncio
from collections import defaultdict
from urllib.parse import urljoin
from typing import Dict, Tuple

import pydash
import requests
import voluptuous as vol
from num2words import num2words

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
    KEY_COMMAND,
    KEY_COMMANDS,
    KEY_COMMAND_TEMPLATE,
    KEY_COMMAND_TEMPLATES,
    KEY_DATA,
    KEY_DATA_TEMPLATE,
    KEY_INCLUDE,
    KEY_EXCLUDE,
    KEY_DOMAINS,
    KEY_ENTITIES,
    KEY_REGEX,
)

from .conversation import RhasspyConversationAgent
from .core import command_to_sentences
from .default_commands import DEFAULT_INTENTS
from .intent_handlers import (
    DeviceStateIntent,
    SetTimerIntent,
    TimerReadyIntent,
    TriggerAutomationIntent,
    TriggerAutomationLaterIntent,
    make_state_handler,
)


# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# Base URL of Rhasspy web API
CONF_API_URL = "api_url"

# Language to use for generating default utterances
CONF_LANGUAGE = "language"

# User defined commands by intent
CONF_INTENTS = "intents"

# User defined slots and values
CONF_SLOTS = "slots"

# User defined words and pronunciations
CONF_CUSTOM_WORDS = "custom_words"

# Name replacements for entities
CONF_NAME_REPLACE = "name_replace"

# If True, Rhasspy conversation agent is registered
CONF_REGISTER_CONVERSATION = "register_conversation"

# List of intents for Rhasspy to handle
CONF_HANDLE_INTENTS = "handle_intents"

# Speech responses for intent handling
CONF_RESPONSE_TEMPLATES = "reponse_templates"

# State names for question intents (e.g., "on" for INTENT_IS_DEVICE_ON)
CONF_INTENT_STATES = "intent_states"

# Seconds before re-training occurs after new component loaded
CONF_TRAIN_TIMEOUT = "train_timeout"

# Default settings
DEFAULT_API_URL = "http://localhost:12101/api"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SLOTS = {}
DEFAULT_CUSTOM_WORDS = {}
DEFAULT_REGISTER_CONVERSATION = True
DEFAULT_TRAIN_TIMEOUT = 1.0
DEFAULT_NAME_REPLACE = {
    # English
    # Replace dashes/underscores with spaces
    "en-US": {KEY_REGEX: [{r"[_-]": " "}]},
    #
    # French
    # Split dashed words (est-ce -> est -ce)
    # Replace dashes with spaces
    "fr-FR": {KEY_REGEX: [{r"-": " -"}, {r"_": " "}]},
}

DEFAULT_HANDLE_INTENTS = [
    INTENT_IS_DEVICE_ON,
    INTENT_IS_DEVICE_OFF,
    INTENT_IS_COVER_OPEN,
    INTENT_IS_COVER_CLOSED,
    INTENT_DEVICE_STATE,
    INTENT_TRIGGER_AUTOMATION,
    INTENT_TRIGGER_AUTOMATION_LATER,
    INTENT_SET_TIMER,
    INTENT_TIMER_READY,
]

DEFAULT_RESPONSE_TEMPLATES = {
    "en-US": {
        INTENT_IS_DEVICE_ON: T(
            "{{ 'Yes' if entity.state in states else 'No' }}. {{ entity.name }} {{ 'are' if entity.name.endswith('s') else 'is' }} on."
        ),
        INTENT_IS_DEVICE_OFF: T(
            "{{ 'Yes' if entity.state in states else 'No' }}. {{ entity.name }} {{ 'are' if entity.name.endswith('s') else 'is' }} off."
        ),
        INTENT_IS_COVER_OPEN: T(
            "{{ 'Yes' if entity.state in states else 'No' }}. {{ entity.name }} {{ 'are' if entity.name.endswith('s') else 'is' }} open."
        ),
        INTENT_IS_COVER_CLOSED: T(
            "{{ 'Yes' if entity.state in states else 'No' }}. {{ entity.name }} {{ 'are' if entity.name.endswith('s') else 'is' }} closed."
        ),
        INTENT_DEVICE_STATE: T(
            "{{ entity.name }} {% 'are' if entity.name.endswith('s') else 'is' %} {{ entity.state }}."
        ),
        INTENT_TIMER_READY: T("Timer is ready."),
        INTENT_TRIGGER_AUTOMATION: T("Triggered {{ automation.name }}."),
    }
}

DEFAULT_INTENT_STATES = {
    "en-US": {
        INTENT_IS_DEVICE_ON: ["on"],
        INTENT_IS_DEVICE_OFF: ["off"],
        INTENT_IS_COVER_OPEN: ["open"],
        INTENT_IS_COVER_CLOSED: ["closed"],
    }
}

# Config
COMMAND_SCHEMA = vol.Schema(
    {
        vol.Exclusive(KEY_COMMAND, "commands"): str,
        vol.Exclusive(KEY_COMMAND_TEMPLATE, "commands"): cv.template,
        vol.Exclusive(KEY_COMMANDS, "commands"): vol.All(cv.ensure_list, [str]),
        vol.Exclusive(KEY_COMMAND_TEMPLATES, "commands"): vol.All(
            cv.ensure_list, [cv.template]
        ),
        vol.Optional(KEY_DATA): vol.Schema({str: object}),
        vol.Optional(KEY_DATA_TEMPLATE): vol.Schema({str: cv.template}),
        vol.Optional(KEY_INCLUDE): vol.Schema(
            {vol.Optional(KEY_DOMAINS): vol.All(cv.ensure_list, [str])},
            {vol.Optional(KEY_ENTITIES): vol.All(cv.ensure_list, [cv.entity_id])},
        ),
        vol.Optional(KEY_EXCLUDE): vol.Schema(
            {vol.Optional(KEY_ENTITIES): vol.All(cv.ensure_list, [cv.entity_id])}
        ),
        vol.Optional(CONF_HANDLE_INTENTS, DEFAULT_HANDLE_INTENTS): vol.Schema(
            cv.ensure_list, [str]
        ),
        vol.Optional(CONF_RESPONSE_TEMPLATES): vol.Schema({str: cv.template}),
    }
)

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.All(
            {
                vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): vol.All(
                    str, vol.In(SUPPORT_LANGUAGES)
                ),
                vol.Optional(CONF_API_URL, default=DEFAULT_API_URL): cv.url,
                vol.Optional(
                    CONF_REGISTER_CONVERSATION, default=DEFAULT_REGISTER_CONVERSATION
                ): bool,
                vol.Optional(CONF_SLOTS, DEFAULT_SLOTS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str])}
                ),
                vol.Optional(CONF_CUSTOM_WORDS, DEFAULT_CUSTOM_WORDS): vol.Schema(
                    {str: str}
                ),
                vol.Optional(CONF_INTENTS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str, COMMAND_SCHEMA])}
                ),
                vol.Optional(CONF_NAME_REPLACE): {
                    vol.Optional(KEY_REGEX, {}): vol.All(
                        cv.ensure_list, [vol.Schema({str: str})]
                    ),
                    vol.Optional(KEY_ENTITIES, {}): vol.Schema({cv.entity_id: str}),
                },
                vol.Optional(CONF_INTENT_STATES): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str])}
                ),
                vol.Optional(CONF_TRAIN_TIMEOUT, DEFAULT_TRAIN_TIMEOUT): float,
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
    api_url = conf.get(CONF_API_URL, DEFAULT_API_URL)

    register_conversation = conf.get(
        CONF_REGISTER_CONVERSATION, DEFAULT_REGISTER_CONVERSATION
    )

    if register_conversation:
        # Register converation agent
        agent = RhasspyConversationAgent(hass, api_url)
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
        self.api_url: str = config.get(CONF_API_URL, DEFAULT_API_URL)

        # URL to POST sentences.ini
        self.sentences_url: str = urljoin(self.api_url, "sentences")

        # URL to POST custom_words.txt
        self.custom_words_url: str = urljoin(self.api_url, "custom-words")

        # URL to POST slots
        self.slots_url: str = urljoin(self.api_url, "slots")

        # URL to train profile
        self.train_url: str = urljoin(self.api_url, "train")

        # e.g., en-US
        self.language: str = config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        # entity id -> [entity state, clean name]
        self.entities: Dict[str, Tuple[State, str]] = {}

        # lower-cased entity names in self.entities
        self.entity_names_lower: Set[str] = set()

        # Language-specific settings
        self.name_replace = self._get_for_language(
            CONF_NAME_REPLACE, DEFAULT_NAME_REPLACE
        )

        # Threads/events
        self.train_thread = None
        self.train_event = threading.Event()

        self.train_timer_thread = None
        self.train_timer_event = threading.Event()
        self.train_timer_seconds = self.config.get(
            CONF_TRAIN_TIMEOUT, DEFAULT_TRAIN_TIMEOUT
        )

    # -------------------------------------------------------------------------

    async def async_initialize(self):
        """Initialize Rhasspy provider."""

        # Get intent responses
        response_templates = dict(
            DEFAULT_RESPONSE_TEMPLATES.get(
                self.language, DEFAULT_RESPONSE_TEMPLATES[DEFAULT_LANGUAGE]
            )
        )
        if CONF_RESPONSE_TEMPLATES in self.config:
            for intent_obj, template in self.config[CONF_RESPONSE_TEMPLATES].items():
                # Overwrite default
                response_templates[intent_obj] = template

        # Get states of intents
        intent_states = dict(
            DEFAULT_INTENT_STATES.get(
                self.language, DEFAULT_INTENT_STATES[DEFAULT_LANGUAGE]
            )
        )
        if CONF_INTENT_STATES in self.config:
            for intent_obj, states in self.config[CONF_INTENT_STATES].items():
                intent_states[intent_obj] = states

        # Register intent handlers
        handle_intents = set(
            self.config.get(CONF_HANDLE_INTENTS, DEFAULT_HANDLE_INTENTS)
        )

        if INTENT_DEVICE_STATE in handle_intents:
            intent.async_register(
                self.hass, DeviceStateIntent(response_templates[INTENT_DEVICE_STATE])
            )

        # Generate handlers for specific states (on, open, etc.)
        for state_intent in [
            INTENT_IS_DEVICE_ON,
            INTENT_IS_DEVICE_OFF,
            INTENT_IS_COVER_OPEN,
            INTENT_IS_COVER_CLOSED,
        ]:
            if state_intent in handle_intents:
                intent.async_register(
                    self.hass,
                    make_state_handler(
                        state_intent,
                        intent_states[state_intent],
                        response_templates[state_intent],
                    ),
                )

        if INTENT_SET_TIMER in handle_intents:
            intent.async_register(self.hass, SetTimerIntent())

        if INTENT_TIMER_READY in handle_intents:
            intent.async_register(
                self.hass, TimerReadyIntent(response_templates[INTENT_TIMER_READY])
            )

        if INTENT_TRIGGER_AUTOMATION in handle_intents:
            intent.async_register(
                self.hass,
                TriggerAutomationIntent(response_templates[INTENT_TRIGGER_AUTOMATION]),
            )

        if INTENT_TRIGGER_AUTOMATION_LATER in handle_intents:
            intent.async_register(self.hass, TriggerAutomationLaterIntent())

        # Register for component loaded event
        self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, self.component_loaded)

    @callback
    def component_loaded(self, event):
        """Handle a new component loaded."""
        old_entity_count = len(self.entities)

        # User-defined entity names for speech to text
        entity_name_map = self.name_replace.get(KEY_ENTITIES, {})

        # Regex replacements for cleaning names
        name_regexes = self.name_replace.get(KEY_REGEX, {})

        # Language used for num2words (en-US -> en_US)
        num2words_lang = self.language.replace("-", "_")

        for state in self.hass.states.async_all():
            # Skip entities that have already been loaded
            if state.entity_id in self.entities:
                continue

            if state.entity_id in entity_name_map:
                # User-defined name
                entity_name = entity_name_map[state.entity_id]
            else:
                # Try to clean name
                entity_name = state.name

                # Do number replacement
                words = re.split(r"\s+", entity_name)
                for i, word in enumerate(words):
                    try:
                        number = float(word)
                        try:
                            words[i] = num2words(number, lang=num2words_lang)
                        except NotImplementedError:
                            # Use default language (U.S. English)
                            words[i] = num2words(number)
                    except ValueError:
                        pass

                entity_name = " ".join(words)

                # Do regex replacements
                for replacements in name_regexes:
                    for pattern, replacement in replacements.items():
                        entity_name = re.sub(pattern, replacement, entity_name)

            # Ensure we don't duplicate names
            entity_name_lower = entity_name.lower()
            if entity_name_lower not in self.entity_names_lower:
                self.entities[state.entity_id] = (state, entity_name)
                self.entity_names_lower.add(entity_name_lower)

        # Detemine if new entities have been added
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

                # Override defaults with user commands
                intent_commands = dict(
                    DEFAULT_INTENTS.get(
                        self.language, DEFAULT_INTENTS[DEFAULT_LANGUAGE]
                    )
                )
                for intent_type, commands in self.config.get(CONF_INTENTS, {}).items():
                    intent_commands[intent_type] = commands

                # Generate commands
                for intent_type, commands in intent_commands.items():
                    for command in commands:
                        for sentence in command_to_sentences(
                            self.hass, command, self.entities
                        ):
                            sentences_by_intent[intent_type].append(sentence)

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
                                    sentence = f"\\{sentence}"

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

    # -------------------------------------------------------------------------

    def _get_for_language(self, config_key, default_values):
        """Gets language-specific values for a configuration option."""
        if config_key in self.config:
            # User-specified value
            return self.cofig[config_key]

        # Try current language, fall back to default language (U.S. English)
        return default_values.get(self.language, default_values[DEFAULT_LANGUAGE])


# -----------------------------------------------------------------------------
