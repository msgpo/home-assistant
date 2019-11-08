"""Const for Rhasspy integration."""

DOMAIN = "rhasspy"

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

# Custom intents
INTENT_IS_DEVICE_ON = "IsDeviceOn"
INTENT_IS_DEVICE_OFF = "IsDeviceOff"

INTENT_IS_COVER_OPEN = "IsCoverOpen"
INTENT_IS_COVER_CLOSED = "IsCoverClosed"

INTENT_DEVICE_STATE = "DeviceState"

INTENT_TRIGGER_AUTOMATION = "TriggerAutomation"
INTENT_TRIGGER_AUTOMATION_LATER = "TriggerAutomationLater"

INTENT_SET_TIMER = "SetTimer"

INTENT_TIMER_READY = "TimerReady"
