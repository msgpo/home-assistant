#!/usr/bin/env python3
import io
import subprocess
import wave
import logging
from typing import Iterable, Set

import pydash

from homeassistant.helpers.template import Template
from homeassistant.components.stt import SpeechMetadata

_LOGGER = logging.getLogger("rhasspy")

# -----------------------------------------------------------------------------
# Audio Functions
# -----------------------------------------------------------------------------


def maybe_convert_audio(metadata: SpeechMetadata, audio_data: bytes) -> bytes:
    """Converts audio data to 16-bit, 16Khz mono."""
    rate = int(metadata.sample_rate)
    width = int(metadata.bit_rate)

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


# -----------------------------------------------------------------------------
# Voice Command Functions
# -----------------------------------------------------------------------------


def command_to_sentences(hass, command, entities, template_dict={}) -> Iterable[str]:
    if isinstance(command, str):
        # Literal sentence
        yield command
    elif isinstance(command, Template):
        # Template sentence
        command.hass = hass
        yield command.render(template_dict)
    else:
        # Command object
        # - command
        #   command_template
        #   commands
        #   command_templates
        #   data
        #   data_template
        #   include:
        #     domains
        #     entities
        #   exclude:
        #     domains
        #     entities
        commands = []
        have_templates = False

        if "command" in command:
            commands = [command["command"]]
        elif "commands" in command:
            commands = command["commands"]
        elif "command_template" in command:
            commands = [command["command_template"]]
            have_templates = True
        elif "command_templates" in command:
            commands = command["command_templates"]
            have_templates = True

        possible_entity_ids: Set[str] = set()
        if have_templates:
            # Gather all entities to be used in command templates
            if "include" in command:
                include_domains = set(pydash.get(command, "include.domains", []))
                for entity_id, state in entities.items():
                    if state.domain in include_domains:
                        possible_entity_ids.add(entity_id)

                include_entities = pydash.get(command, "include.entities", [])
                possible_entity_ids.update(include_entities)

            if "exclude" in command:
                exclude_entities = pydash.get(command, "exclude.entities", [])
                possible_entity_ids.difference_update(exclude_entities)

        # Generate Rhasspy sentences for each command (template)
        for sub_command in commands:
            if not have_templates:
                # Literal sentence
                command_strs = [sub_command]
            elif len(possible_entity_ids) == 0:
                # Assume template doesn't refer to entities
                command_strs = command_to_sentences(hass, sub_command, entities)
            else:
                # Render template for each possible entity (state)
                command_strs = []
                for entity_id in possible_entity_ids:
                    state = entities.get(entity_id)
                    if state is not None:
                        template_dict = {"entity": state}
                        command_strs.extend(
                            command_to_sentences(
                                hass, sub_command, entities, template_dict=template_dict
                            )
                        )

            # Extra data to attach to command
            command_data = dict(command.get("data", {}))
            command_data_template = command.get("data_template", {})

            # Render templates
            for data_key, data_template in command_data_template.items():
                data_template.hass = hass
                command_data[data_key] = data_template.render()

            # Append to sentences.
            # Use special form "(:){key:value}" to carry
            # information with the voice command without
            # changing to wording.
            for command_str in command_strs:
                for data_key, data_value in command_data.items():
                    data_value = str(data_value)
                    command_str = f"{command_str} (:){{{data_key}:{data_value}}}"

                yield command_str
