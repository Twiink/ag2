# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from ag2 import Agent, Context, MemoryStream, ToolResult
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultEvent
from ag2.testing import TestConfig
from ag2.tools import LocalEnvironment, SandboxCodeTool, SandboxShellTool


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["shell", "code"])
@pytest.mark.parametrize(
    "description",
    ["Run a command", "Run in {placeholder}", 'Return JSON {"status": "ok"} in {placeholder}', "Print a literal {"],
)
async def test_sandbox_description_preserves_literal_braces(kind: str, description: str, tmp_path: Path) -> None:
    environment = LocalEnvironment(tmp_path)
    stream = MemoryStream()
    results: list[ToolResult] = []
    stream.where(ToolResultEvent).subscribe(lambda event: results.append(event.result))
    sandbox: SandboxShellTool | SandboxCodeTool
    if kind == "shell":
        description = description.replace("{placeholder}", "{workdir}")
        sandbox = SandboxShellTool(environment, description=description)
        expected = description.replace("{workdir}", str(tmp_path))
        arguments = {"command": f'"{sys.executable}" -c "print(42)"'}
    else:
        description = description.replace("{placeholder}", "{languages}")
        sandbox = SandboxCodeTool(environment, description=description)
        expected = description.replace("{languages}", "python, bash")
        arguments = {"code": "print(42)", "language": "python"}
    [schema] = await sandbox.schemas(Context(stream))
    assert schema.function.description == expected
    config = TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(name=sandbox.name, arguments=json.dumps(arguments))])),
        "done",
    )
    agent = Agent("sandbox", config=config, tools=[sandbox])

    reply = await agent.ask("Run the example", stream=stream)

    assert reply.body == "done"
    assert results == [ToolResult("42")]
