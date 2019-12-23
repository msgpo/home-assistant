"""
Support for Rhasspy voice assistant integration.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
import logging

import voluptuous as vol

from homeassistant.components.conversation import async_set_agent
import homeassistant.helpers.config_validation as cv

from .const import DOMAIN
from .conversation import RhasspyConversationAgent

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger(__name__)

# Config
CONF_API_URL = "api_url"
CONF_REGISTER_CONVERSATION = "register_conversation"

DEFAULT_REGISTER_CONVERSATION = True

CONFIG_SCHEMA = vol.Schema(
    {
        DOMAIN: vol.All(
            {
                vol.Required(CONF_API_URL): cv.url,
                vol.Optional(
                    CONF_REGISTER_CONVERSATION, default=DEFAULT_REGISTER_CONVERSATION
                ): bool,
            }
        )
    },
    extra=vol.ALLOW_EXTRA,
)

# -----------------------------------------------------------------------------


async def async_setup(hass, config):
    """Set up Rhasspy integration."""
    conf = config.get(DOMAIN)
    if conf is None:
        # Don't initialize
        return True

    # Load configuration
    api_url = conf[CONF_API_URL]
    if not api_url.endswith("/"):
        api_url = api_url + "/"
        conf[CONF_API_URL] = api_url

    # Register conversation agent
    register_conversation = conf.get(
        CONF_REGISTER_CONVERSATION, DEFAULT_REGISTER_CONVERSATION
    )

    if register_conversation:
        agent = RhasspyConversationAgent(hass, api_url)
        async_set_agent(hass, agent)
        _LOGGER.debug("Registered Rhasspy conversation agent")

    return True
