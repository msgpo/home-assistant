"""Support for Rhasspy speech to text."""
import io
import os
import time
import asyncio
import logging
import shutil
from typing import List

import aiohttp
import voluptuous as vol
import requests
import homeassistant.helpers.config_validation as cv

from homeassistant.components.stt import Provider, SpeechMetadata, SpeechResult
from homeassistant.components.stt.const import (
    AudioFormats,
    AudioCodecs,
    AudioBitRates,
    AudioSampleRates,
    SpeechResultState,
)

from .const import SUPPORT_LANGUAGES
from .core import maybe_convert_audio, buffer_to_wav
from .command import WebRTCVADCommandListener, VoiceCommandResult

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)


# Use webrtcvad to detect when voice command ends
CONF_DETECT_SILENCE = "detect_silence"

# True if audio should be streamed directly to Rhasspy (HTTPAudioRecorder)
CONF_STREAM = "stream"

# URL to stream audio to
CONF_STREAM_URL = "stream_url"

# Size of audio data chunk to read from stream
CONF_CHUNK_SIZE = "chunk_size"

# Default settings
DEFAULT_DETECT_SILENCE = True
DEFAULT_STREAM = True
DEFAULT_STREAM_URL = "http://localhost:12333"
DEFAULT_CHUNK_SIZE = 960

# Config
PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_DETECT_SILENCE, default=DEFAULT_DETECT_SILENCE): bool,
        vol.Optional(CONF_STREAM, default=DEFAULT_STREAM): bool,
        vol.Optional(CONF_STREAM_URL, default=DEFAULT_STREAM_URL): cv.url,
        vol.Optional(CONF_CHUNK_SIZE, default=DEFAULT_CHUNK_SIZE): cv.positive_int,
    }
)

# -----------------------------------------------------------------------------


async def async_get_engine(hass, config):
    """Set up Rhasspy speech to text component."""
    provider = RhasspySTTProvider(hass, config)
    _LOGGER.info("Loaded Rhasspy stt provider")
    return provider


# -----------------------------------------------------------------------------


class RhasspySTTProvider(Provider):
    """Rhasspy speech to text provider."""

    def __init__(self, hass, conf):
        self.hass = hass
        self.config = conf

        # Check if sox is available for WAV conversion
        self.sox_available: bool = len(shutil.which("sox")) > 0

        # True if audio should be streamed
        self.do_stream = conf.get(CONF_STREAM, DEFAULT_STREAM)

        # URL to stream microphone audio
        self.stream_url = conf.get(CONF_STREAM_URL, DEFAULT_STREAM_URL)

        # True if voice2json should wait until silence to do STT
        self.detect_silence = conf.get(CONF_DETECT_SILENCE, DEFAULT_DETECT_SILENCE)

        # Number of bytes to send at once
        self.chunk_size = conf.get(CONF_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)

        # Used to detect speech/silence
        self.listener = WebRTCVADCommandListener() if self.detect_silence else None

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: aiohttp.StreamReader
    ) -> SpeechResult:
        """Process an audio stream to STT service.

        Only streaming of content are allow!
        """

        _LOGGER.debug("Receiving audio")
        text_result = ""

        try:
            if self.do_stream:
                # Stream remotely
                _LOGGER.debug(f"Streaming audio to {self.stream_url}")

                async def chunk_generator():
                    async for audio_chunk in stream.iter_chunked(self.chunk_size):
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
                except aiohttp.ClientOSError:
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

                async for audio_chunk in stream.iter_chunked(self.chunk_size):
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
        except Exception as e:
            _LOGGER.error("async_process_audio_stream")

        return SpeechResult(result=SpeechResultState.ERROR)

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
    def supported_bit_rates(self) -> List[AudioBitRates]:
        """Return a list of supported bitrates."""
        if self.sox_available:
            # Can support all bitrates
            return list(AudioBitRates)
        else:
            # Only 16-bit
            return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> List[AudioSampleRates]:
        """Return a list of supported samplerates."""
        if self.sox_available:
            # Can support any sample rate
            return list(AudioSampleRates)
        else:
            # Only 16Khz
            return [AudioSampleRates.SAMPLERATE_16000]
