#!/usr/bin/env python3
import sys
import threading
import asyncio
import argparse
import concurrent.futures

import aiohttp

# Streams 16-bit 16Khz mono audio from stdin to Rhasspy STT endpoint.
# Prints result.
#
# Example:
# arecord -r 16000 -c 1 -f S16_LE -t raw | ./mic_stream.py --token "..."

# -----------------------------------------------------------------------------

loop = asyncio.get_event_loop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://localhost:8123/api/stt/rhasspy",
        help="URL of Rhasspy STT endpoint",
    )
    parser.add_argument("--token", help="Authorization token")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=960,
        help="Number of bytes to read/send at a time",
    )
    args = parser.parse_args()

    try:
        loop.run_until_complete(do_stream(args.url, args.token, args.chunk_size))
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


async def stdin_generator(chunk_size):
    while True:
        chunk = sys.stdin.buffer.read(chunk_size)
        yield chunk


async def do_stream(url, token=None, chunk_size=960):
    headers = {
        "X-Speech-Content": "format=wav; codec=pcm; sample_rate=16000; bit_rate=16; language=en-US"
    }

    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, headers=headers, data=stdin_generator(chunk_size), chunked=True
        ) as resp:
            print(await resp.text())


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
