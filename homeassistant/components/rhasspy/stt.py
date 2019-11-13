"""
Support for Rhasspy speech to text.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
import asyncio
import io
import logging
import os
from pathlib import Path
import shutil
import time
from typing import List
from urllib.parse import urljoin
import wave

import aiohttp
import requests
import voluptuous as vol

from homeassistant.components.stt import Provider, SpeechMetadata, SpeechResult
from homeassistant.components.stt.const import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechResultState,
)
import homeassistant.helpers.config_validation as cv

from .const import SUPPORT_LANGUAGES

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# URL to POST WAV audio to
CONF_SPEECH_URL = "speech_url"

# Default settings
DEFAULT_SPEECH_URL = "http://localhost:12101/api/speech-to-text"

# Config
PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({vol.Optional(CONF_SPEECH_URL): cv.url})

# -----------------------------------------------------------------------------


async def async_get_engine(hass, config, discovery_info):
    """Set up Rhasspy speech to text component."""
    provider = RhasspySTTProvider(hass, config)
    _LOGGER.debug("Loaded Rhasspy stt provider")
    return provider


# -----------------------------------------------------------------------------


class RhasspySTTProvider(Provider):
    """Rhasspy speech to text provider."""

    def __init__(self, hass, conf):
        self.hass = hass
        self.config = conf

        # URL to stream microphone audio
        self.speech_url = conf.get(CONF_SPEECH_URL, None)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: aiohttp.StreamReader
    ) -> SpeechResult:
        """Process an audio stream to STT service.

        Only streaming of content are allow!
        """

        if self.speech_url is None:
            # Try to get API URL from Rhasspy provider
            provider = self.hass.data.get(DOMAIN)
            if provider is not None:
                self.speech_url = urljoin(provider.api_url, "/speech-to-text")
            else:
                # Use default
                self.speech_url = DEFAULT_SPEECH_URL

        _LOGGER.debug("Receiving audio")
        text_result = ""

        try:
            # First chunk is a WAV header (no frames)
            header_chunk = True
            with io.BytesIO() as wav_io:
                wav_file = wave.open(wav_io, "wb")
                async for audio_chunk, _ in stream.iter_chunks():
                    if header_chunk:
                        # Extract WAV information
                        header_chunk = False
                        with io.BytesIO(audio_chunk) as header_io:
                            with wave.open(header_io) as header_file:
                                wav_file.setnchannels(header_file.getnchannels())
                                wav_file.setsampwidth(header_file.getsampwidth())
                                wav_file.setframerate(header_file.getframerate())
                    else:
                        # Everything after first chunk is audio data.
                        # Add to in-memory WAV file.
                        wav_file.writeframes(audio_chunk)

                wav_file.close()
                wav_data = wav_io.getvalue()
                _LOGGER.debug(f"Received {len(wav_data)} byte(s)")

            # POST to Rhasspy endpoint
            headers = {"Content-Type": "audio/wav"}
            text_result = requests.post(
                self.speech_url, data=wav_data, headers=headers
            ).text
            _LOGGER.debug(text_result)

            return SpeechResult(text=text_result, result=SpeechResultState.SUCCESS)
        except Exception as e:
            _LOGGER.exception("async_process_audio_stream")

        return SpeechResult(text="", result=SpeechResultState.ERROR)

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
        return list(AudioBitRates)

    @property
    def supported_sample_rates(self) -> List[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return list(AudioSampleRates)

    @property
    def supported_channels(self) -> List[AudioChannels]:
        """Return a list of supported channels."""
        return list(AudioChannels)
