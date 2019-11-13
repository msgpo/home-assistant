"""
Rhasspy agent for conversation integration.

For more details about this integration, please refer to the documentation at
https://home-assistant.io/integrations/rhasspy/
"""
from abc import ABC, abstractmethod
import logging
from urllib.parse import urljoin

import aiohttp
import pydash

from homeassistant import core
from homeassistant.helpers import intent

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class RhasspyConversationAgent(ABC):
    """Rhasspy conversation agent."""

    def __init__(self, hass: core.HomeAssistant, api_url: str):
        """Initialize the conversation agent."""
        self.hass = hass
        self.intent_url = urljoin(api_url, "text-to-intent")

    async def async_process(self, text: str) -> intent.IntentResponse:
        """Process a sentence."""
        _LOGGER.debug(f"Processing '{text}' ({self.intent_url})")

        async with aiohttp.ClientSession() as session:
            params = {"nohass": "true"}
            async with session.post(self.intent_url, data=text, params=params) as resp:
                result = await resp.json()
                intent_type = pydash.get(result, "intent.name", "")
                if len(intent_type) > 0:
                    _LOGGER.debug(result)

                    text = result.get("raw_text", result.get("text", ""))
                    slots = result.get("slots", {})

                    return await intent.async_handle(
                        self.hass,
                        DOMAIN,
                        intent_type,
                        {key: {"value": value} for key, value in slots.items()},
                        text,
                    )
                else:
                    _LOGGER.warning("Received empty intent")
