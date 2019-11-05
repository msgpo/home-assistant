#!/usr/bin/env python3
import logging

_LOGGER = logging.getLogger("rhasspy")

import sys
import argparse
import math
import threading
import time
from collections import deque
from typing import List, BinaryIO, TextIO, Optional
from enum import Enum

import webrtcvad

# -----------------------------------------------------------------------------


class VoiceCommandResult(Enum):
    """Result of processing one audio chunk."""

    INCOMPLETE = 0
    COMPLETE = 1
    TIMEOUT = 2


class VoiceCommandState(Enum):
    """State of voice command."""

    BEFORE_COMMAND = 0
    IN_COMMAND = 1
    AFTER_COMMAND = 2


# -----------------------------------------------------------------------------


class WebRTCVADCommandListener:
    """Buffers spoken audio until silence using webrtcvad."""

    def __init__(
        self,
        vad_mode: int = 3,
        sample_rate: int = 16000,
        chunk_size: int = 960,
        min_seconds: float = 2,
        max_seconds: float = 30,
        speech_seconds: float = 0.3,
        silence_seconds: float = 0.5,
        before_seconds: float = 0.25,
    ):

        # Aggressiveness of voice activity detection (VAD)
        self.vad_mode: int = vad_mode

        # Hz
        self.sample_rate: int = sample_rate

        # Size of audio chunks in bytes (must be 10, 20, 30 ms)
        self.chunk_size: int = chunk_size
        self.min_seconds: float = min_seconds
        self.max_seconds: float = max_seconds
        self.speech_seconds: float = speech_seconds
        self.silence_seconds: float = silence_seconds
        self.before_seconds: float = before_seconds

        # Verify settings
        assert self.vad_mode in range(1, 4), f"VAD mode must be 1-3 (got {vad_mode})"

        chunk_ms = 1000 * ((self.chunk_size / 2) / self.sample_rate)
        assert chunk_ms in [10, 20, 30], (
            "Sample rate and chunk size must make for 10, 20, or 30 ms buffer sizes,"
            + f" assuming 16-bit mono at {sample_rate} Hz audio (got {chunk_ms} ms)"
        )

        # Voice detector
        self.vad: webrtcvad.Vad = webrtcvad.Vad()
        self.vad.set_mode(self.vad_mode)

        self.seconds_per_buffer: float = self.chunk_size / self.sample_rate

        # Store some number of seconds of audio data immediately before voice command starts
        self.before_buffers: int = int(
            math.ceil(self.before_seconds / self.seconds_per_buffer)
        )
        self.before_command_chunks: int = deque(maxlen=self.before_buffers)

        # Store audio data during voice command
        self.command_buffer: bytes = bytes()

        # Pre-compute values
        self.speech_buffers: int = int(
            math.ceil(self.speech_seconds / self.seconds_per_buffer)
        )

        # Set initial state
        self.start_command()

    def start_command(self):
        """Resets voice command listener."""
        self.state: VoiceCommandState = VoiceCommandState.BEFORE_COMMAND

        # Audio data for voice command
        self.command_buffer: bytes = bytes()

        # Maximum number of chunks before timeout
        self.max_buffers: int = int(
            math.ceil(self.max_seconds / self.seconds_per_buffer)
        )

        # Minimum number of chunks a voice command can have
        self.min_command_buffers: int = int(
            math.ceil(self.min_seconds / self.seconds_per_buffer)
        )

        # Number of chunks left before voice command starts
        self.speech_buffers_left: int = self.speech_buffers

        # True when webrtcvad detected speech in most recent chunk
        self.is_speech: bool = False

        # True if voice command has started and not stopped
        self.in_command: bool = False

        # True after voice command has stopped
        self.after_command: bool = False

        # Total number of seconds of audio recording
        self.current_seconds: float = 0

    def process_audio(self, audio_chunk: bytes) -> VoiceCommandResult:
        """Processes a single chunk of audio data."""
        if self.in_command:
            # Add to current voice command
            self.command_buffer += audio_chunk
        else:
            # Cache chunk in case voice command starts
            self.before_command_chunks.append(audio_chunk)

        self.current_seconds += self.seconds_per_buffer

        # Check maximum number of seconds to record
        self.max_buffers -= 1
        if self.max_buffers <= 0:
            return VoiceCommandResult.TIMEOUT

        result: VoiceCommandResult = VoiceCommandResult.INCOMPLETE

        # Detect speech in chunk
        self.is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)

        # Handle state changes
        if self.is_speech and self.speech_buffers_left > 0:
            # Must detect speech in a certain number of chunks before starting command
            self.speech_buffers_left -= 1
        elif self.is_speech and not self.in_command:
            # Start of voice command (unless min seconds is not reached)
            self.state = VoiceCommandState.IN_COMMAND
            self.in_command = True
            self.after_command = False
            self.min_command_buffers = int(
                math.ceil(self.min_seconds / self.seconds_per_buffer)
            )
        elif self.in_command and (self.min_command_buffers > 0):
            # In voice command, before minimum seconds
            self.min_command_buffers -= 1
        elif not self.is_speech:
            # Outside of speech
            if not self.in_command:
                # Reset
                self.speech_buffers_left = self.speech_buffers
            elif self.after_command and (self.silence_buffers > 0):
                # After command, before stop
                self.silence_buffers -= 1
            elif self.after_command and (self.silence_buffers <= 0):
                # Command complete
                result = VoiceCommandResult.COMPLETE
            elif self.in_command and (self.min_command_buffers <= 0):
                # Transition to after command
                self.state = VoiceCommandState.AFTER_COMMAND
                self.after_command = True
                self.silence_buffers = int(
                    math.ceil(self.silence_seconds / self.seconds_per_buffer)
                )

        return result

    def get_audio(self):
        """Gets audio from recorded voice command."""
        before_buffer: bytes = bytes()
        for chunk in self.before_command_chunks:
            before_buffer += chunk

        return before_buffer + self.command_buffer
