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


# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# Base URL of Rhasspy web server
CONF_WEB_URL = "web_url"

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

# Default settings
DEFAULT_WEB_URL = "http://localhost:12101"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SLOTS = {}
DEFAULT_CUSTOM_WORDS = {}
DEFAULT_REGISTER_CONVERSATION = True
DEFAULT_NAME_REPLACE = {KEY_REGEX: [{"[_-]": " "}]}

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

# DEFAULT_INTENT_RESPONSES: {INTENT_IS_DEVICE_ON: "{% %}"}

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
                vol.Optional(CONF_SLOTS, DEFAULT_SLOTS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str])}
                ),
                vol.Optional(CONF_CUSTOM_WORDS, DEFAULT_CUSTOM_WORDS): vol.Schema(
                    {str: str}
                ),
                vol.Optional(CONF_INTENTS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str, COMMAND_SCHEMA])}
                ),
                vol.Optional(CONF_NAME_REPLACE, DEFAULT_NAME_REPLACE): {
                    vol.Optional(KEY_REGEX, {}): vol.All(
                        cv.ensure_list, [vol.Schema({str: str})]
                    ),
                    vol.Optional(KEY_ENTITIES, {}): vol.Schema({cv.entity_id: str}),
                },
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

        # entity id -> [entity state, clean name]
        self.entities: Dict[str, Tuple[State, str]] = {}

        # lower-cased entity names in self.entities
        self.entity_names_lower: Set[str] = set()

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
        name_replace = self.config.get(CONF_NAME_REPLACE, DEFAULT_NAME_REPLACE)
        entity_name_map = name_replace.get(KEY_ENTITIES, {})
        name_regexes = name_replace.get(KEY_REGEX, {})
        num2words_lang = self.language.replace("-", "_")

        for state in self.hass.states.async_all():
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
                if CONF_INTENTS in self.config:
                    # User-provided commands
                    intent_commands = self.config[CONF_INTENTS]
                else:
                    # Default commands
                    intent_commands = DEFAULT_INTENTS.get(
                        self.language, DEFAULT_INTENTS.get(DEFAULT_LANGUAGE)
                    )

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


# -----------------------------------------------------------------------------
