"""Support for voice2json speech to text."""
import io
import os
import asyncio
import logging
import shutil
import subprocess
import wave
import json
import time
import tempfile
import platform
import threading
from typing import List, Union, TextIO, BinaryIO
from enum import Enum
from pathlib import Path

import pydash
import aiohttp
import voluptuous as vol
import homeassistant.helpers.config_validation as cv
from homeassistant.core import Event, callback
from homeassistant.const import (
    EVENT_HOMEASSISTANT_STOP,
    EVENT_COMPONENT_LOADED,
    ATTR_FRIENDLY_NAME,
)
from homeassistant.helpers import intent
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

from .core import async_download, maybe_convert_audio, buffer_to_wav

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# See http://voice2json.org/#supported-languages
# Excluding untested profiles for now.
PROFILE_LANGUAGES = {
    "en-us_pocketsphinx-cmu": "en-US",
    "en-us_kaldi-zamia": "en-US",
    "nl_pocketsphinx-cmu": "nl-NL",
    "nl_kaldi-cgn": "nl-NL",
    "fr_pocketsphinx-cmu": "fr-FR",
    "de_pocketsphinx-cmu": "de-DE",
    "de_kaldi-zamia": "de-DE",
    "el_pocketsphinx-cmu": "el-GR",
    "it_pocketsphinx-cmu": "it-IT",
    "pt-br_pocketsphinx-cmu": "pt-BR",
    "ru_pocketsphinx-cmu": "ru-RU",
    "es_pocketsphinx-cmu": "es-ES",
    "sv_kaldi-monreal": "sv-SV",
    "vi_kaldi-montreal": "vi-VI",
}

SUPPORT_LANGUAGES = sorted(list(set(PROFILE_LANGUAGES.values())))

# Name of voice2json profile, e.g. en-us_pocketsphinx-cmu
CONF_PROFILE_NAME = "profile_name"

# URL format string for where to download profile.
# Takes {profile_name} as a format argument.
CONF_PROFILE_URL_TEMPLATE = "profile_url_template"

# URL format string for where to download voice2json.
# Takes {machine} as a format argument (from platform.machine()).
CONF_VOICE2JSON_URL_TEMPLATE = "voice2json_url_template"

# One of "closed" or "open".
# closed: use only utterances from sentences.ini (best)
# open: use general language model (okay)
CONF_TRANSCRIPTION_MODE = "transcription_mode"

# Use webrtcvad to detect when voice command ends
CONF_DETECT_SILENCE = "detect_silence"

# Automatically generate turn on/off and toggle utterances for all entities with
# friendly names.
CONF_DEFAULT_UTTERANCES = "default_utterances"

# Default settings
DEFAULT_PROFILE_NAME = "en-us_pocketsphinx-cmu"
DEFAULT_PROFILE_URL_TEMPLATE = (
    "https://github.com/synesthesiam/{profile_name}/archive/v1.0.tar.gz"
)
DEFAULT_VOICE2JSON_URL_TEMPLATE = "https://github.com/synesthesiam/voice2json/releases/download/v1.0/voice2json_1.0_{machine}.tar.gz"

# TODO: Add mixed mode
DEFAULT_TRANSCRIPTION_MODE = "closed"

DEFAULT_DETECT_SILENCE = True
DEFAULT_DEFAULT_UTTERANCES = True

# Size of audio data chunk to read from audio stream
CHUNK_SIZE = 960

# Config schema
PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_PROFILE_NAME, default=DEFAULT_PROFILE_NAME): cv.string,
        vol.Optional(
            CONF_PROFILE_URL_TEMPLATE, default=DEFAULT_PROFILE_URL_TEMPLATE
        ): cv.string,
        vol.Optional(
            CONF_VOICE2JSON_URL_TEMPLATE, default=DEFAULT_VOICE2JSON_URL_TEMPLATE
        ): cv.string,
        vol.Optional(
            CONF_TRANSCRIPTION_MODE, default=DEFAULT_TRANSCRIPTION_MODE
        ): vol.All(cv.string, vol.In(["open", "closed"])),
        vol.Optional(CONF_DETECT_SILENCE, default=DEFAULT_DETECT_SILENCE): bool,
        vol.Optional(CONF_DEFAULT_UTTERANCES, default=DEFAULT_DEFAULT_UTTERANCES): bool,
    }
)

# x86_64, armhf (armv7l), aarch64
CPU_ARCH = {"armv7l": "armhf"}.get(platform.machine(), platform.machine())

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
    """Set up voice2json speech to text component."""
    provider = Voice2JsonSTTProvider(hass, config)
    await provider.async_initialize()

    # Schedule shutdown on stop event
    hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STOP, provider.async_shutdown)

    return provider


# -----------------------------------------------------------------------------


class Voice2JsonSTTProvider(Provider):
    """voice2json speech to text provider."""

    def __init__(self, hass, conf):
        # Check if sox is available for WAV conversion
        self.hass = hass
        self.config = conf

        # True if sox can be used to convert WAV files
        self.sox_available: bool = len(shutil.which("sox")) > 0

        # Name of profile, e.g. en-us_pocketsphinx-cmu
        self.profile_name = conf.get(CONF_PROFILE_NAME, DEFAULT_PROFILE_NAME)

        # en-US, etc.
        self.language = PROFILE_LANGUAGES[self.profile_name]

        # True if voice2json should wait until silence to do STT
        self.detect_silence = conf.get(CONF_DETECT_SILENCE, DEFAULT_DETECT_SILENCE)

        # voice2json print-profile
        self.voice2json_profile = {}

        # config/voice2json
        self.voice2json_dir = Path(hass.config.path("voice2json"))

        # Directory with profile.yml
        self.profile_dir = self.voice2json_dir / "profiles" / self.profile_name

        # Path to voice2json executable script
        self.voice2json_exe = self.voice2json_dir / "voice2json" / "bin" / "voice2json"

        # Path to voice2json base directory
        self.voice2json_dir = self.voice2json_dir / "voice2json" / "lib" / "voice2json"

        # closed, open, or mixed
        self.transcription_mode = conf.get(
            CONF_TRANSCRIPTION_MODE, DEFAULT_TRANSCRIPTION_MODE
        )

        # subprocess to do transcriptions
        self.transcribe_proc = None

        # entity id -> friendly name
        self.entities = {}

        # Function to upper/lower case words
        self.word_transform = lambda w: w

        # Threads/events
        self.download_event = threading.Event()

        self.train_thread = None
        self.train_event = threading.Event()

        self.train_timer_thread = None
        self.train_timer_event = threading.Event()
        self.train_timer_seconds = 1

    async def async_initialize(self):
        """Initialize provider."""
        # Fix word casing according to profile
        word_casing = pydash.get(
            self.voice2json_profile, "training.word-casing", "ignore"
        )
        if word_casing == "upper":
            self.word_transform = lambda w: w.upper()
        elif word_casing == "lower":
            self.word_transform = lambda w: w.lower()

        # Register for component loaded event
        self.hass.bus.async_listen(EVENT_COMPONENT_LOADED, self.component_loaded)

        # Make sure voice2json/profile are downloaded
        await self.async_download_dependencies()

        # Get profile
        self.voice2json_profile = json.load(self.voice2json("print-profile"))

    @callback
    async def async_shutdown(self, event: Event):
        """Shut down the transcription process."""
        if self.transcribe_proc is not None:
            self.transcribe_proc.terminate()
            self.transcribe_proc = None

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: aiohttp.StreamReader
    ) -> SpeechResult:
        """Process an audio stream to STT service.

        Only streaming of content are allow!
        """

        record_proc = None
        if self.detect_silence:
            # Start subprocess to detect silence
            record_proc = self.voice2json(
                "record-command", "--audio-source", "-", text=False, stream=True
            )

        try:
            # Buffer with all incoming audio
            all_audio = bytes()

            # Buffer with voice command audio only
            speech_audio = bytes()

            # Process chunks
            async for audio_chunk in stream.iter_chunked(CHUNK_SIZE):
                if self.sox_available:
                    # Convert to 16-bit 16Khz mono
                    audio_chunk = maybe_convert_audio(metadata, audio_chunk)

                # _LOGGER.info(len(audio_chunk))
                all_audio += audio_chunk
                if record_proc is not None:
                    # Send chunk to voice2json for silence detection
                    record_proc.stdin.write(audio_chunk)
                    record_proc.stdin.flush()

                    # Check if voice command is finished
                    record_proc.poll()
                    if record_proc.returncode is not None:
                        # Read audio with voice command
                        speech_audio = record_proc.stdout.read()
                        break
                    else:
                        # Use entire audio stream as a backup
                        speech_audio = all_audio
        finally:
            # Stop silence detection
            if record_proc is not None:
                record_proc.terminate()

        # Wrap up in WAV structure
        wav_data = buffer_to_wav(all_audio)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="wb") as temp_file:
            temp_file.write(wav_data)

            # Send temp file name to voice2json
            self.transcribe_proc.stdin.write((temp_file.name + "\n").encode())
            self.transcribe_proc.stdin.flush()

            # Read result
            json_result = json.loads(self.transcribe_proc.stdout.readline().decode())
            _LOGGER.debug(json_result)

        return SpeechResult(text=json_result["text"], result=SpeechResultState.SUCCESS)

    # -------------------------------------------------------------------------

    @callback
    def component_loaded(self, event):
        """Handle a new component loaded."""
        old_entity_count = len(self.entities)
        for state in self.hass.states.async_all():
            friendly_name = state.attributes.get(ATTR_FRIENDLY_NAME)
            if friendly_name is not None:
                self.entities[state.entity_id] = self.word_transform(friendly_name)

        if len(self.entities) > old_entity_count:
            _LOGGER.info("Need to retrain profile")
            self.schedule_retrain()

    # TODO: Make this use asyncio.subprocess
    def voice2json(
        self, *args, stream=False, text=True, input=None, stderr=None
    ) -> Union[TextIO, BinaryIO, subprocess.Popen]:
        """Calls voice2json with the given arguments and current profile."""

        # Set up subprocess environment variables
        env = os.environ.copy()
        env["voice2json_dir"] = str(self.voice2json_dir)
        env["PATH"] = str(self.voice2json_dir / "bin") + ":" + env["PATH"]

        # voice2json --profile <PROFILE> command <ARGS>
        command = [self.voice2json_exe, "--profile", str(self.profile_dir)] + [
            str(a) for a in args
        ]

        _LOGGER.debug(command)

        if stream:
            # Keep voice2json process running
            if text:
                # Text-based I/O
                return subprocess.Popen(
                    command,
                    env=env,
                    universal_newlines=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )
            else:
                # Binary I/O
                return subprocess.Popen(
                    command,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                )
        else:
            # Finish voice2json process and return file object
            if text:
                # Text-based I/O
                return io.StringIO(
                    subprocess.check_output(
                        command,
                        env=env,
                        universal_newlines=True,
                        input=input,
                        stderr=stderr,
                    )
                )
            else:
                # Binary I/O
                return io.BytesIO(
                    subprocess.check_output(
                        command, env=env, input=input, stderr=stderr
                    )
                )

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
            # Make sure voice2json/profile are downloaded
            self.download_event.wait()

            # Wait for re-train request
            self.train_event.wait()
            self.train_event.clear()

            # stt.voice2json.intents
            intent_utterances = self.config.get("intents", {})

            # Add turn on/off and toggle for all entities
            if self.config.get(CONF_DEFAULT_UTTERANCES, DEFAULT_DEFAULT_UTTERANCES):
                default_utterances = DEFAULT_UTTERANCES.get(self.language)
                if default_utterances is None:
                    # Use English utterances if translations aren't available
                    default_utterances = DEFAULT_UTTERANCES["en-US"]

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
                # Generate custom sentences.ini
                sentences_path = self.profile_dir / "sentences.ini"
                with open(sentences_path, "w") as sentences_file:
                    for intent_type, utterances in intent_utterances.items():
                        print(f"[{intent_type}]", file=sentences_file)

                        for utterance in utterances:
                            if utterance.startswith("["):
                                # Escape "[" at start
                                utterance = f"\\{utterance}"

                            print(utterance, file=sentences_file)

                        print("", file=sentences_file)
            else:
                # Use sentences.ini without modifications
                _LOGGER.warn("Using sentences.ini as-is")

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
                _LOGGER.debug(f"Writing custom words ({custom_words_path})")

                with open(custom_words_path, "w") as custom_words_file:
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

            # Check for slots
            slots = self.config.get("slots", {})
            if len(slots) > 0:
                slots_dir = Path(
                    pydash.get(
                        self.voice2json_profile,
                        "training.slots-directory",
                        self.profile_dir / "slots",
                    )
                )

                _LOGGER.debug(f"Writing slots ({slots_dir})")
                slots_dir.mkdir(parents=True, exist_ok=True)

                for slot_name, slot_values in slots.items():
                    # Accept either string or list of strings
                    if isinstance(slot_values, str):
                        slot_values = [slot_values]

                    with open(slots_dir / slot_name, "w") as slot_file:
                        # One value per line
                        for value in slot_values:
                            print(value.strip(), file=slot_file)

            # Train profile
            _LOGGER.info("Training profile")
            self.voice2json(
                "train-profile", "--db-file", str(self.profile_dir / ".doit.db")
            )

            # (Re)-start transcription process
            transcribe_args = []
            if self.transcription_mode == "open":
                transcribe_args.append("--open")

            if self.transcribe_proc is not None:
                self.transcribe_proc.terminate()
                self.transcribe_proc = None

            _LOGGER.debug("Starting transcription process")
            self.transcribe_proc = self.voice2json(
                "transcribe-wav",
                "--stdin-files",
                *transcribe_args,
                stream=True,
                text=False,
            )

            _LOGGER.info("Ready")

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

    async def async_download_dependencies(self):
        """Downloads and extracts voice2json and profile."""
        download_dir = self.voice2json_dir / "download"

        # Check for voice2json
        voice2json_base_dir = self.voice2json_dir / "voice2json"
        if not voice2json_base_dir.exists():
            voice2json_base_dir.mkdir(parents=True)

            _LOGGER.debug(f"Missing voice2json ({voice2json_base_dir})")
            voice2json_tar = download_dir / "voice2json.tar.gz"

            if not voice2json_tar.exists():
                url = self.config.get(
                    CONF_VOICE2JSON_URL_TEMPLATE, DEFAULT_VOICE2JSON_URL_TEMPLATE
                ).format(machine=CPU_ARCH)
                _LOGGER.info(f"Need to download voice2json ({url})")

                await async_download(url, voice2json_tar)

            _LOGGER.info(f"Extracting {voice2json_tar} to {voice2json_base_dir}")
            subprocess.check_call(
                ["tar", "-C", str(voice2json_base_dir), "-xzf", str(voice2json_tar)]
            )

        # Check for profile
        if not self.profile_dir.exists():
            self.profile_dir.mkdir(parents=True)

            _LOGGER.debug(f"Missing profile ({self.profile_dir})")
            profile_tar = download_dir / f"{self.profile_name}.tar.gz"

            if not profile_tar.exists():
                url = self.config.get(
                    CONF_PROFILE_URL_TEMPLATE, DEFAULT_PROFILE_URL_TEMPLATE
                ).format(profile_name=self.profile_name)
                _LOGGER.info(f"Need to download profile ({url})")

                await async_download(url, profile_tar)

            _LOGGER.debug(f"Extracting {profile_tar} to {self.profile_dir}")
            subprocess.check_call(
                ["tar", "-C", str(self.profile_dir), "-xzf", str(profile_tar)]
            )

        # Signal that voice2json and profile are ready
        self.download_event.set()

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
