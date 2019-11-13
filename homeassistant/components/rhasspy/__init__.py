"""
Support for Rhasspy voice assistant integration.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
import asyncio
from collections import defaultdict
import io
import logging
import re
import threading
import time
from typing import Dict, Tuple
from urllib.parse import urljoin

from num2words import num2words
import pydash
import requests
import voluptuous as vol

from homeassistant.components.conversation import async_set_agent
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER
from homeassistant.components.light import INTENT_SET
from homeassistant.components.shopping_list import INTENT_ADD_ITEM, INTENT_LAST_ITEMS
from homeassistant.const import ATTR_FRIENDLY_NAME, EVENT_COMPONENT_LOADED
from homeassistant.core import Event, State, callback
from homeassistant.helpers import intent
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.template import Template as T
import homeassistant.util.color as color_util

from .const import (
    DOMAIN,
    INTENT_DEVICE_STATE,
    INTENT_IS_COVER_CLOSED,
    INTENT_IS_COVER_OPEN,
    INTENT_IS_DEVICE_OFF,
    INTENT_IS_DEVICE_ON,
    INTENT_IS_DEVICE_STATE,
    INTENT_SET_TIMER,
    INTENT_TIMER_READY,
    INTENT_TRIGGER_AUTOMATION,
    INTENT_TRIGGER_AUTOMATION_LATER,
    KEY_COMMAND,
    KEY_COMMAND_TEMPLATE,
    KEY_COMMAND_TEMPLATES,
    KEY_COMMANDS,
    KEY_DATA,
    KEY_DATA_TEMPLATE,
    KEY_DOMAINS,
    KEY_ENTITIES,
    KEY_EXCLUDE,
    KEY_INCLUDE,
    KEY_REGEX,
    SUPPORT_LANGUAGES,
)
from .conversation import RhasspyConversationAgent
from .core import EntityCommandInfo, command_to_sentences
from .default_commands import DEFAULT_INTENT_COMMANDS
from .intent_handlers import (
    DeviceStateIntent,
    IsDeviceStateIntent,
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
CONF_INTENT_COMMANDS = "intent_commands"

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

# List of possible items that can be added to the shopping list
CONF_SHOPPING_LIST_ITEMS = "shopping_list_items"

# If True, generate default voice commands
CONF_MAKE_INTENT_COMMANDS = "make_intent_commands"

# Default settings
DEFAULT_API_URL = "http://localhost:12101/api"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_SLOTS = {
    "light_color": [
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "pink",
        "purple",
        "violet",
        "red",
        "yellow",
        "orange",
        "white",
    ]
}
DEFAULT_CUSTOM_WORDS = {}
DEFAULT_REGISTER_CONVERSATION = True
DEFAULT_TRAIN_TIMEOUT = 1.0
DEFAULT_SHOPPING_LIST_ITEMS = []
DEFAULT_MAKE_INTENT_COMMANDS = True

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
    INTENT_IS_DEVICE_STATE,
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
        INTENT_IS_DEVICE_STATE: T(
            "{{ 'Yes' if entity.state == state else 'No' }}. {{ entity.name }} {{ 'are' if entity.name.endswith('s') else 'is' }} {{ state.replace('_', ' ') }}."
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
                vol.Optional(CONF_SLOTS): vol.Schema(
                    {str: vol.All(cv.ensure_list, [str])}
                ),
                vol.Optional(CONF_CUSTOM_WORDS, DEFAULT_CUSTOM_WORDS): vol.Schema(
                    {str: str}
                ),
                vol.Optional(CONF_INTENT_COMMANDS): vol.Schema(
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
                vol.Optional(
                    CONF_SHOPPING_LIST_ITEMS, DEFAULT_SHOPPING_LIST_ITEMS
                ): vol.All(cv.ensure_list, [str]),
                vol.Optional(
                    CONF_MAKE_INTENT_COMMANDS, default=DEFAULT_MAKE_INTENT_COMMANDS
                ): vol.Any(
                    bool,
                    vol.Schema(
                        {
                            vol.Exclusive(
                                KEY_INCLUDE, CONF_MAKE_INTENT_COMMANDS
                            ): vol.All(cv.ensure_list, [str]),
                            vol.Exclusive(
                                KEY_EXCLUDE, CONF_MAKE_INTENT_COMMANDS
                            ): vol.All(cv.ensure_list, [str]),
                        }
                    ),
                ),
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
    if not api_url.endswith("/"):
        api_url = api_url + "/"
        conf[CONF_API_URL] = api_url

    register_conversation = conf.get(
        CONF_REGISTER_CONVERSATION, DEFAULT_REGISTER_CONVERSATION
    )

    if register_conversation:
        # Register converation agent
        agent = RhasspyConversationAgent(hass, api_url)
        async_set_agent(hass, agent)

        _LOGGER.debug("Registered Rhasspy conversation agent")

    provider = RhasspyProvider(hass, conf)
    await provider.async_initialize()

    hass.data[DOMAIN] = provider

    # Register services
    async def async_train_handle(service):
        """Service handle for train."""
        _LOGGER.debug("Re-training profile")
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

        # entity_id -> EntityCommandInfo
        self.entities: Dict[str, EntityCommandInfo] = {}

        # Language-specific name replacements
        self.name_replace = dict(
            DEFAULT_NAME_REPLACE.get(
                self.language, DEFAULT_NAME_REPLACE[DEFAULT_LANGUAGE]
            )
        )
        for key, value in self.config.get(CONF_NAME_REPLACE, {}).items():
            # Overwrite with user settings
            self.name_replace[key] = value

        # Regex replacements for cleaning names
        self.name_regexes = self.name_replace.get(KEY_REGEX, {})

        # Language used for num2words (en-US -> en_US)
        self.num2words_lang = self.language.replace("-", "_")
        if self.language == "sv-SV":
            # Use Danish numbers, since Swedish is not supported.
            self.num2words_lang = "dk"

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
            for intent_name, template in self.config[CONF_RESPONSE_TEMPLATES].items():
                # Overwrite default
                response_templates[intent_name] = template

        # Get states of intents
        intent_states = dict(
            DEFAULT_INTENT_STATES.get(
                self.language, DEFAULT_INTENT_STATES[DEFAULT_LANGUAGE]
            )
        )
        if CONF_INTENT_STATES in self.config:
            for intent_name, states in self.config[CONF_INTENT_STATES].items():
                intent_states[intent_name] = states

        # Register intent handlers
        handle_intents = set(
            self.config.get(CONF_HANDLE_INTENTS, DEFAULT_HANDLE_INTENTS)
        )

        if INTENT_IS_DEVICE_STATE in handle_intents:
            intent.async_register(
                self.hass,
                IsDeviceStateIntent(response_templates[INTENT_IS_DEVICE_STATE]),
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

        # Generate default slots.
        # Numbers zero to one hundred.
        number_0_100 = []
        for number in range(0, 101):
            try:
                number_str = num2words(number, lang=self.num2words_lang)
            except NotImplementedError:
                # Use default language (U.S. English)
                number_str = num2words(number)

            # Clean up dashes, etc.
            number_str = self._clean_name(number_str)

            # Add substitutions, so digits will show up downstream instead of
            # words.
            words = re.split(r"\s+", number_str)
            for i, word in enumerate(words):
                if i == 0:
                    words[i] = f"{word}:{number}"
                else:
                    words[i] = word + ":"

            number_0_100.append(" ".join(words))

        DEFAULT_SLOTS["number_0_100"] = number_0_100

        # Register for component loaded event
        self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, self.component_loaded)

    @callback
    def component_loaded(self, event):
        """Handle a new component loaded."""
        old_entity_count = len(self.entities)

        # User-defined entity names for speech to text
        entity_name_map = self.name_replace.get(KEY_ENTITIES, {})

        for state in self.hass.states.async_all():
            # Skip entities that have already been loaded
            if state.entity_id in self.entities:
                continue

            if state.entity_id in entity_name_map:
                # User-defined name
                speech_name = entity_name_map[state.entity_id]
            else:
                # Try to clean name
                speech_name = self._clean_name(state.name)

            # Clean name but don't replace numbers.
            # This should be matched by intent.async_match_state.
            friendly_name = state.name.replace("_", " ")

            info = EntityCommandInfo(
                entity_id=state.entity_id,
                state=state,
                speech_name=speech_name,
                friendly_name=friendly_name,
            )

            self.entities[state.entity_id] = info

        # Detemine if new entities have been added
        if len(self.entities) > old_entity_count:
            _LOGGER.debug("Need to retrain profile")
            self.schedule_retrain()

    def _clean_name(self, name: str, replace_numbers=True) -> str:
        # Do number replacement
        words = re.split(r"\s+", name)

        if replace_numbers:
            # Convert numbers to words
            for i, word in enumerate(words):
                try:
                    number = float(word)
                    try:
                        words[i] = num2words(number, lang=self.num2words_lang)
                    except NotImplementedError:
                        # Use default language (U.S. English)
                        words[i] = num2words(number)
                except ValueError:
                    pass

        name = " ".join(words)

        # Do regex replacements
        for replacements in self.name_regexes:
            for pattern, replacement in replacements.items():
                name = re.sub(pattern, replacement, name)

        return name

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
        """Re-trains Rhasspy. Works with a timer to avoid too many re-trains."""
        while True:
            # Wait for re-train request
            self.train_event.wait()
            self.train_event.clear()

            try:
                sentences_by_intent: Dict[str, List[str]] = defaultdict(list)
                make_default_commands = self.config.get(
                    CONF_MAKE_INTENT_COMMANDS, DEFAULT_MAKE_INTENT_COMMANDS
                )

                if make_default_commands:
                    # Use default commands
                    intent_commands = dict(
                        DEFAULT_INTENT_COMMANDS.get(
                            self.language, DEFAULT_INTENT_COMMANDS[DEFAULT_LANGUAGE]
                        )
                    )

                    # Determine if intents should be included or excluded
                    if isinstance(make_default_commands, dict):
                        if KEY_INCLUDE in make_default_commands:
                            include_intents = set(make_default_commands[KEY_INCLUDE])
                            intent_commands = {
                                intent_name: commands
                                for intent_name, commands in intent_commands.items()
                                if intent_name in include_intents
                            }
                        elif KEY_EXCLUDE in make_default_commands:
                            for intent_name in make_default_commands[KEY_EXCLUDE]:
                                del intent_commands[intent_name]
                else:
                    # No default commands
                    intent_commands = {}

                # Override defaults with user commands
                for intent_type, commands in self.config.get(
                    CONF_INTENT_COMMANDS, {}
                ).items():
                    intent_commands[intent_type] = commands

                # Generate commands
                for intent_type, commands in intent_commands.items():
                    if intent_type == INTENT_ADD_ITEM:
                        # Special case for shopping list
                        sentences_by_intent[intent_type].extend(
                            self._get_shopping_list_commands(
                                intent_commands.get(INTENT_ADD_ITEM, [])
                            )
                        )
                    else:
                        # All other intents
                        for command in commands:
                            for sentence in command_to_sentences(
                                self.hass, command, self.entities
                            ):
                                sentences_by_intent[intent_type].append(sentence)

                # Prune empty intents
                for intent_type in list(sentences_by_intent):
                    sentences = [
                        s
                        for s in sentences_by_intent[intent_type]
                        if len(s.strip()) > 0
                    ]
                    if len(sentences) > 0:
                        sentences_by_intent[intent_type] = sentences
                    else:
                        del sentences_by_intent[intent_type]

                # Check for custom sentences
                num_sentences = sum(len(s) for s in sentences_by_intent.values())
                if num_sentences > 0:
                    _LOGGER.debug(f"Writing sentences ({self.sentences_url})")

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
                else:
                    _LOGGER.warning("No commands generated. Not overwriting sentences.")

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
                slots = dict(DEFAULT_SLOTS)
                for slot_name, slot_values in self.config.get(CONF_SLOTS, {}).items():
                    slots[slot_name] = slot_values

                if len(slots) > 0:
                    _LOGGER.debug(f"Writing slots ({self.slots_url})")
                    for slot_name, slot_values in list(slots.items()):
                        # Accept either string or list of strings
                        if isinstance(slot_values, str):
                            slots[slot_name] = [slot_values]

                    # POST slots (JSON)
                    requests.post(self.slots_url, json=slots)

                # Train profile
                _LOGGER.debug(f"Training profile ({self.train_url})")
                requests.post(self.train_url)

                _LOGGER.debug("Ready")
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

    def _get_shopping_list_commands(self, commands):
        """Generate voice commands for possible shopping list items."""
        possible_items = self.config.get(
            CONF_SHOPPING_LIST_ITEMS, DEFAULT_SHOPPING_LIST_ITEMS
        )

        # Generate clean item names
        item_names = {
            item_name: self._clean_name(item_name) for item_name in possible_items
        }

        for command in commands:
            if KEY_COMMAND_TEMPLATE in command:
                templates = [command[KEY_COMMAND_TEMPLATE]]
            elif KEY_COMMAND_TEMPLATES in command:
                templates = command[KEY_COMMAND_TEMPLATES]
            else:
                # No templates
                continue

            for item_name, clean_item_name in item_names.items():
                for template in templates:
                    template.hass = self.hass
                    yield template.async_render(
                        {"item_name": item_name, "clean_item_name": clean_item_name}
                    )
