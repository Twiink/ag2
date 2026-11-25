# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.response import response_schema
from ag2.testing import TestConfig


class NumberParser:
    """Double a parsed integer."""

    def __call__(self, content: str) -> int:
        return int(content) * 2


class AsyncNumberParser:
    """Double a parsed integer asynchronously."""

    async def __call__(self, content: str) -> int:
        return int(content) * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("parser_type", [NumberParser, AsyncNumberParser])
@pytest.mark.parametrize("sync_to_thread", [False, True])
@pytest.mark.parametrize("name", [None, "double"])
async def test_callable_instance_response(
    parser_type: type[NumberParser] | type[AsyncNumberParser], sync_to_thread: bool, name: str | None
) -> None:
    schema = response_schema(parser_type(), name=name, sync_to_thread=sync_to_thread)
    assert schema.name == (name or parser_type.__name__)
    assert schema.description == parser_type.__doc__
    agent = Agent("parser", config=TestConfig("21"), response_schema=schema)
    reply = await agent.ask("Double the number")
    assert await reply.content() == 42
