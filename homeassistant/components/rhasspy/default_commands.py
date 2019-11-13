"""
Templates for auto-generated Rhasspy voice commands.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
from homeassistant.components.cover import INTENT_CLOSE_COVER, INTENT_OPEN_COVER
from homeassistant.components.light import INTENT_SET
from homeassistant.components.shopping_list import INTENT_ADD_ITEM, INTENT_LAST_ITEMS
from homeassistant.helpers import intent
from homeassistant.helpers.template import Template as T

from .const import (
    INTENT_DEVICE_STATE,
    INTENT_IS_COVER_CLOSED,
    INTENT_IS_COVER_OPEN,
    INTENT_IS_DEVICE_OFF,
    INTENT_IS_DEVICE_ON,
    INTENT_SET_TIMER,
    INTENT_TRIGGER_AUTOMATION,
    INTENT_TRIGGER_AUTOMATION_LATER,
    KEY_COMMAND_TEMPLATES,
    KEY_COMMANDS,
    KEY_DATA,
    KEY_DATA_TEMPLATE,
    KEY_DOMAINS,
    KEY_ENTITIES,
    KEY_EXCLUDE,
    KEY_INCLUDE,
)

# Default command templates by intent.
# Includes built-in Home Assistant intents as well as Rhasspy intents.
DEFAULT_INTENT_COMMANDS = {
    "en-US": {
        intent.INTENT_TURN_ON: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light", "switch"],
                    KEY_ENTITIES: ["group.all_lights", "group.all_switches"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "turn on [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"
                    ),
                    T(
                        "turn [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} on"
                    ),
                ],
            }
        ],
        intent.INTENT_TURN_OFF: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light", "switch"],
                    KEY_ENTITIES: ["group.all_lights", "group.all_switches"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "turn off [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"
                    ),
                    T(
                        "turn [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} off"
                    ),
                ],
            }
        ],
        intent.INTENT_TOGGLE: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light", "switch"],
                    KEY_ENTITIES: ["group.all_lights", "group.all_switches"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "toggle [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"
                    ),
                    T(
                        "[the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} toggle"
                    ),
                ],
            }
        ],
        INTENT_ADD_ITEM: [
            {
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "add [the|a|an] ({{ clean_item_name }}){name:{{ item_name }}} to [my] shopping list"
                    )
                ]
            }
        ],
        INTENT_LAST_ITEMS: [
            {
                KEY_COMMANDS: [
                    "what is on my shopping list",
                    "[list | tell me] [my] shopping list [items]",
                ]
            }
        ],
        INTENT_OPEN_COVER: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["cover"],
                    KEY_ENTITIES: ["group.all_covers"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T("open [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"),
                    T("[the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} open"),
                ],
            }
        ],
        INTENT_CLOSE_COVER: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["cover"],
                    KEY_ENTITIES: ["group.all_covers"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T("close [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"),
                    T("[the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} close"),
                ],
            }
        ],
        INTENT_SET: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light"],
                    KEY_ENTITIES: ["group.all_lights"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "set [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} [to] ($light_color){color}"
                    ),
                    T(
                        "set [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} [brightness [to] | to brightness] ($number_0_100){brightness}"
                    ),
                    T(
                        "set [the] brightness [of] [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} [to] ($number_0_100){brightness}"
                    ),
                    T(
                        "set [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} [to] (maximum){brightness:100} brightness"
                    ),
                ],
            }
        ],
        INTENT_DEVICE_STATE: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: [
                        "light",
                        "switch",
                        "binary_sensor",
                        "sensor",
                        "cover",
                    ],
                    KEY_ENTITIES: [
                        "group.all_lights",
                        "group.all_switches",
                        "group.all_covers",
                    ],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "what (is | are) [the|a|an] (state | states) of [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}}"
                    ),
                    T(
                        "what [is | are] [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} (state | states)"
                    ),
                ],
            }
        ],
        INTENT_IS_DEVICE_ON: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light", "switch"],
                    KEY_ENTITIES: ["group.all_lights", "group.all_switches"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(is | are) [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} on"
                    )
                ],
            }
        ],
        INTENT_IS_DEVICE_OFF: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["light", "switch"],
                    KEY_ENTITIES: ["group.all_lights", "group.all_switches"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(is | are) [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} off"
                    )
                ],
            }
        ],
        INTENT_IS_COVER_OPEN: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["cover"],
                    KEY_ENTITIES: ["group.all_covers"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(is | are) [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} open"
                    )
                ],
            }
        ],
        INTENT_IS_COVER_CLOSED: [
            {
                KEY_INCLUDE: {
                    KEY_DOMAINS: ["cover"],
                    KEY_ENTITIES: ["group.all_covers"],
                },
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(is | are) [the|a|an] ({{ speech_name }}){name:{{ friendly_name }}} closed"
                    )
                ],
            }
        ],
        INTENT_TRIGGER_AUTOMATION: [
            {
                KEY_INCLUDE: {KEY_DOMAINS: ["automation"]},
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(run | execute | trigger) [program | automation] ({{ speech_name }}){name:{{ friendly_name }}}"
                    )
                ],
            }
        ],
        INTENT_TRIGGER_AUTOMATION_LATER: [
            {
                KEY_INCLUDE: {KEY_DOMAINS: ["automation"]},
                KEY_COMMAND_TEMPLATES: [
                    T(
                        "(run | execute | trigger) [program | automation] ({{ speech_name }}){name:{{ friendly_name }}} (in | after) <SetTimer.time_expr>"
                    ),
                    T(
                        "(in | after) <SetTimer.time_expr> (run | trigger) [program | automation] ({{ speech_name }}){name:{{ friendly_name }}}"
                    ),
                ],
            }
        ],
        INTENT_SET_TIMER: [
            {
                KEY_COMMANDS: [
                    "two_to_nine = (two:2 | three:3 | four:4 | five:5 | six:6 | seven:7 | eight:8 | nine:9)",
                    "one_to_nine = (one:1 | <two_to_nine>)",
                    "",
                    "teens = (ten:10 | eleven:11 | twelve:12 | thirteen:13 | fourteen:14 | fifteen:15 | sixteen:16 | seventeen:17 | eighteen:18 | nineteen:19)",
                    "",
                    "tens = (twenty:20 | thirty:30 | forty:40 | fifty:50)",
                    "one_to_nine = (one:1 | <two_to_nine>)",
                    "",
                    "one_to_fifty_nine = (<one_to_nine> | <teens> | <tens> [<one_to_nine>])",
                    "two_to_fifty_nine = (<two_to_nine> | <teens> | <tens> [<one_to_nine>])",
                    "hour_half_expr = (<one_to_nine>{hours} and (a half){{minutes:30}})",
                    "hour_expr = (((one:1){hours}) | ((<one_to_nine>){hours}) | <hour_half_expr>) (hour | hours)",
                    "minute_half_expr = (<one_to_fifty_nine>{minutes} and (a half){{seconds:30}})",
                    "minute_expr = (((one:1){minutes}) | ((<two_to_fifty_nine>){minutes}) | <minute_half_expr>) (minute | minutes)",
                    "second_expr = (((one:1){seconds}) | ((<two_to_fifty_nine>){seconds})) (second | seconds)",
                    "",
                    "time_expr = ((<hour_expr> [[and] <minute_expr>] [[and] <second_expr>]) | (<minute_expr> [[and] <second_expr>]) | <second_expr>)",
                    "",
                    "set [a] timer for <time_expr>",
                ]
            }
        ],
    }
}
