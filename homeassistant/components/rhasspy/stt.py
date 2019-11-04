"""Support for Rhasspy speech to text."""
import io
import os
import time
import asyncio
import logging
import shutil
import threading
from typing import List
from pathlib import Path
from urllib.parse import urljoin

import aiohttp
import voluptuous as vol
import requests
import homeassistant.helpers.config_validation as cv
from homeassistant.core import Event, callback
from homeassistant.helpers import intent
from homeassistant.const import EVENT_COMPONENT_LOADED, ATTR_FRIENDLY_NAME
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER
from homeassistant.components.shopping_list import INTENT_ADD_ITEM, INTENT_LAST_ITEMS

from homeassistant.components.stt import Provider, SpeechMetadata, SpeechResult
from homeassistant.components.stt.const import (
    AudioFormats,
    AudioCodecs,
    AudioBitrates,
    AudioSamplerates,
    SpeechResultState,
)

from .core import maybe_convert_audio, buffer_to_wav
from .command import WebRTCVADCommandListener, VoiceCommandResult

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)


# Base URL of Rhasspy web server
CONF_WEB_URL = "web_url"

# Language to use for generating default utterances
CONF_LANGUAGE = "language"

# Automatically generate turn on/off and toggle utterances for all entities with
# friendly names.
CONF_DEFAULT_UTTERANCES = "default_utterances"

# Use webrtcvad to detect when voice command ends
CONF_DETECT_SILENCE = "detect_silence"

# True if audio should be streamed directly to Rhasspy (HTTPAudioRecorder)
CONF_STREAM = "stream"

# URL to stream audio to
CONF_STREAM_URL = "stream_url"

# Default settings
DEFAULT_WEB_URL = "http://localhost:12101"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_DEFAULT_UTTERANCES = True
DEFAULT_DETECT_SILENCE = True
DEFAULT_STREAM = True
DEFAULT_STREAM_URL = "http://localhost:12333"

SUPPORT_LANGUAGES = [
    "en-US",
    "nl-NL",
    "fr-FR",
    "de-DE",
    "el-GR",
    "it-IT",
    "pt-BR",
    "ru-RU",
    "es-ES",
    "sv-SV",
    "vi-VI",
]

# Size of audio data chunk to read from stream
CHUNK_SIZE = 960

# Config
PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): vol.All(
            cv.string, vol.In(SUPPORT_LANGUAGES)
        ),
        vol.Optional(CONF_WEB_URL, default=DEFAULT_WEB_URL): cv.url,
        vol.Optional(CONF_DEFAULT_UTTERANCES, default=DEFAULT_DEFAULT_UTTERANCES): bool,
        vol.Optional(CONF_DETECT_SILENCE, default=DEFAULT_DETECT_SILENCE): bool,
        vol.Optional(CONF_STREAM, default=DEFAULT_STREAM): bool,
        vol.Optional(CONF_STREAM_URL, default=DEFAULT_STREAM_URL): cv.url,
    }
)

# -----------------------------------------------------------------------------

# TODO: Translate to all supported languages
DEFAULT_UTTERANCES = {
    "en-US": {
        intent.INTENT_TURN_ON: [
            "turn on [the|a|an] {name}",
            "turn [the|a|an] {name} on",
        ],
        intent.INTENT_TURN_OFF: [
            "turn off [the|a|an] {name}",
            "turn [the|a|an] {name} off",
        ],
        intent.INTENT_TOGGLE: ["toggle [the|a|an] {name}", "[the|a|an] {name} toggle"],
    }
}

# -----------------------------------------------------------------------------


async def async_get_engine(hass, config):
    """Set up Rhasspy speech to text component."""
    provider = RhasspySTTProvider(hass, config)
    await provider.async_initialize()

    return provider


# -----------------------------------------------------------------------------


class RhasspySTTProvider(Provider):
    """Rhasspy speech to text provider."""

    def __init__(self, hass, conf):
        self.hass = hass
        self.config = conf

        # Check if sox is available for WAV conversion
        self.sox_available: bool = len(shutil.which("sox")) > 0

        # Base URL of Rhasspy web server
        self.url = conf.get(CONF_WEB_URL, DEFAULT_WEB_URL)

        # URL to POST WAV data
        self.stt_url = urljoin(self.url, "api/speech-to-text")

        # URL to POST sentences.ini
        self.sentences_url = urljoin(self.url, "api/sentences")

        # URL to POST custom_words.txt
        self.custom_words_url = urljoin(self.url, "api/custom-words")

        # URL to POST slots
        self.slots_url = urljoin(self.url, "api/slots")

        # URL to train profile
        self.train_url = urljoin(self.url, "api/train")

        # True if audio should be streamed
        self.do_stream = conf.get(CONF_STREAM, DEFAULT_STREAM)

        # URL to stream microphone audio
        self.stream_url = conf.get(CONF_STREAM_URL, DEFAULT_STREAM_URL)

        # e.g., en-US
        self.language = conf.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)

        # True if voice2json should wait until silence to do STT
        self.detect_silence = conf.get(CONF_DETECT_SILENCE, DEFAULT_DETECT_SILENCE)

        # Used to detect speech/silence
        self.listener = WebRTCVADCommandListener() if self.detect_silence else None

        # entity id -> friendly name
        self.entities = {}

        # Threads/events
        self.train_thread = None
        self.train_event = threading.Event()

        self.train_timer_thread = None
        self.train_timer_event = threading.Event()
        self.train_timer_seconds = 1

    async def async_initialize(self):
        """Initialize provider."""

        # Register for component loaded event
        self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, self.component_loaded)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: aiohttp.StreamReader
    ) -> SpeechResult:
        """Process an audio stream to STT service.

        Only streaming of content are allow!
        """

        text_result = ""

        if self.do_stream:
            # Stream remotely
            _LOGGER.debug(f"Streaming audio to {self.stream_url}")

            async def chunk_generator():
                async for audio_chunk in stream.iter_chunked(CHUNK_SIZE):
                    if self.sox_available:
                        # Convert to 16-bit 16Khz mono
                        audio_chunk = maybe_convert_audio(metadata, audio_chunk)

                    yield audio_chunk

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.stream_url, data=chunk_generator(), chunked=True
                    ) as resp:
                        await resp.text()
            except:
                # Stream was closed
                pass

            text_result = requests.get(self.stream_url).text.strip()
            _LOGGER.info(text_result)
        else:
            # Buffer locally
            if self.detect_silence:
                # Reset VAD
                self.listener.start_command()

            all_audio = bytes()
            speech_audio = all_audio

            async for audio_chunk in stream.iter_chunked(CHUNK_SIZE):
                if self.sox_available:
                    # Convert to 16-bit 16Khz mono
                    audio_chunk = maybe_convert_audio(metadata, audio_chunk)

                all_audio += audio_chunk

                if self.detect_silence:
                    # Process chunk
                    vad_result = self.listener.process_audio(audio_chunk)
                    if vad_result == VoiceCommandResult.COMPLETE:
                        # Command complete
                        speech_audio = self.listener.get_audio()
                        break
                    elif vad_result == VoiceCommandResult.TIMEOUT:
                        break

            # Wrap up in WAV structure
            wav_data = buffer_to_wav(speech_audio)

            # POST to Rhasspy
            headers = {"Content-Type": "audio/wav"}
            text_result = requests.post(
                self.stt_url, data=wav_data, headers=headers
            ).text
            _LOGGER.debug(text_result)

        return SpeechResult(text=text_result, result=SpeechResultState.SUCCESS)

    # -------------------------------------------------------------------------

    @callback
    def component_loaded(self, event):
        """Handle a new component loaded."""
        old_entity_count = len(self.entities)
        for state in self.hass.states.async_all():
            friendly_name = state.attributes.get(ATTR_FRIENDLY_NAME)
            if friendly_name is not None:
                self.entities[state.entity_id] = friendly_name

        if len(self.entities) > old_entity_count:
            _LOGGER.info("Need to retrain profile")
            self.schedule_retrain()

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
                # stt.rhasspy.intents
                intent_utterances = self.config.get("intents", {})

                # Add turn on/off and toggle for all entities
                if self.config.get(CONF_DEFAULT_UTTERANCES, DEFAULT_DEFAULT_UTTERANCES):
                    default_utterances = DEFAULT_UTTERANCES.get(self.language)
                    if default_utterances is None:
                        # Use default (English) utterances if translations aren't available
                        default_utterances = DEFAULT_UTTERANCES[DEFAULT_LANGUAGE]

                    default_intents = [
                        intent.INTENT_TURN_ON,
                        intent.INTENT_TURN_OFF,
                        intent.INTENT_TOGGLE,
                    ]
                    for intent_obj in default_intents:
                        current_utterances = intent_utterances.get(intent_obj, [])

                        # Generate utterance for each entity
                        for entity_id, entity_name in self.entities.items():
                            for utt_format in default_utterances.get(intent_obj, []):
                                current_utterances.append(
                                    utt_format.format(name=entity_name)
                                )

                        if len(current_utterances) > 0:
                            intent_utterances[intent_obj] = current_utterances

                num_utterances = sum(len(u) for u in intent_utterances.values())
                if num_utterances > 0:
                    _LOGGER.debug("Writing sentences ({self.sentences_url})")

                    # Generate custom sentences.ini
                    with io.StringIO() as sentences_file:
                        for intent_type, utterances in intent_utterances.items():
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
                    custom_words_path = Path(
                        pydash.get(
                            self.voice2json_profile,
                            "training.custom-words-file",
                            self.profile_dir / "custom_words.txt",
                        )
                    )
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
                    _LOGGER.info(f"Writing slots ({self.slots_url})")
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

    @property
    def supported_languages(self) -> List[str]:
        """Return a list of supported languages."""
        return SUPPORT_LANGUAGES

    @property
    def supported_formats(self) -> List[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> List[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bitrates(self) -> List[AudioBitrates]:
        """Return a list of supported bitrates."""
        if self.sox_available:
            # Can support all bitrates
            return list(AudioBitrates)
        else:
            # Only 16-bit
            return [AudioBitrates.BITRATE_16]

    @property
    def supported_samplerates(self) -> List[AudioSamplerates]:
        """Return a list of supported samplerates."""
        if self.sox_available:
            # Can support any sample rate
            return list(AudioSamplerates)
        else:
            # Only 16Khz
            return [AudioSamplerates.SAMPLERATE_16000]
