#!/usr/bin/env python3
import io
import subprocess
import wave
import logging

from homeassistant.components.stt import SpeechMetadata

_LOGGER = logging.getLogger("rhasspy")


def maybe_convert_audio(metadata: SpeechMetadata, audio_data: bytes) -> bytes:
    """Converts audio data to 16-bit, 16Khz mono."""
    rate = int(metadata.samplerate)
    width = int(metadata.bitrate)

    # TODO: Check channels
    if (rate == 16000) and (width == 16):
        # No converstion necessary
        return audio_data

    convert_cmd = [
        "sox",
        "-t",
        "raw",
        "-r",
        str(rate),
        "-b",
        str(width),
        "-c",
        str(channels),
        "-",
        "-r",
        "16000",
        "-e",
        "signed-integer",
        "-b",
        "16",
        "-c",
        "1",
        "-t",
        "raw",
        "-",
    ]

    _LOGGER.debug(convert_cmd)

    return subprocess.run(
        convert_cmd, check=True, stdout=subprocess.PIPE, input=audio_data
    ).stdout


def buffer_to_wav(buffer: bytes) -> bytes:
    """Wraps a buffer of raw audio data (16-bit, 16Khz mono) in a WAV"""
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, mode="wb") as wav_file:
            wav_file.setframerate(16000)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(1)
            wav_file.writeframesraw(buffer)

        return wav_buffer.getvalue()
