"""Constants for Rhasspy integration.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
DOMAIN = "rhasspy"

# Language-specific profiles available for Rhasspy.
# See https://github.com/synesthesiam/rhasspy-profiles/releases
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

# Custom intents for Rhasspy integration
INTENT_IS_DEVICE_ON = "IsDeviceOn"
INTENT_IS_DEVICE_OFF = "IsDeviceOff"

INTENT_IS_COVER_OPEN = "IsCoverOpen"
INTENT_IS_COVER_CLOSED = "IsCoverClosed"

INTENT_IS_DEVICE_STATE = "IsDeviceState"
INTENT_DEVICE_STATE = "DeviceState"

INTENT_TRIGGER_AUTOMATION = "TriggerAutomation"
INTENT_TRIGGER_AUTOMATION_LATER = "TriggerAutomationLater"

INTENT_SET_TIMER = "SetTimer"

INTENT_TIMER_READY = "TimerReady"

# Configuration keys
KEY_COMMAND = "command"
KEY_COMMANDS = "commands"
KEY_COMMAND_TEMPLATE = "command_template"
KEY_COMMAND_TEMPLATES = "command_templates"
KEY_DATA = "data"
KEY_DATA_TEMPLATE = "data_template"
KEY_INCLUDE = "include"
KEY_EXCLUDE = "exclude"
KEY_DOMAINS = "domains"
KEY_ENTITIES = "entities"
KEY_REGEX = "regex"
