# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent, MemoryStream, TextInput, ToolResult, events, testing, tool


class Lookup:
    def __call__(self, query: str) -> str:
        return query.upper()


class FalseLookup(Lookup):
    def __bool__(self) -> bool:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lookup", "explicit_name", "call_name"),
    [
        (Lookup(), None, "Lookup"),
        (FalseLookup(), "lookup", "lookup"),
        (Lookup(), "lookup", "lookup"),
    ],
)
async def test_agent_executes_callable_tool(lookup: Lookup, explicit_name: str | None, call_name: str) -> None:
    stream = MemoryStream()
    lookup_tool = tool(lookup, name=explicit_name)
    agent = Agent(
        "lookup-agent",
        config=testing.TestConfig(events.ToolCallEvent(name=call_name, arguments='{"query":"rice"}'), "done"),
        tools=[lookup_tool],
    )

    reply = await agent.ask("Look up rice", stream=stream)

    assert reply.body == "done"
    [batch] = await stream.history.where(events.ToolResultsEvent).all()
    [result] = batch.results
    assert result.result == ToolResult(TextInput("RICE"))
