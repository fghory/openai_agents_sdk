from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, RunConfig, ItemHelpers, MessageOutputItem, trace, TResponseInputItem,
    RunContextWrapper
)
from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import os
import asyncio
import agentops
from dataclasses import dataclass
from typing import Literal

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

agentops.init(AGENTOPS_API_KEY)

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",   
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    # tracing_disabled=True,
)

@dataclass
class CustomContext():
    style: Literal["haiku", "pirate", "robot"]


def custom_instructions(run_context:RunContextWrapper[CustomContext], agent:Agent[CustomContext]):
    context = run_context.context
    if context.style == "haiku":
        return "Only respond in haikus."
    elif context.style == "pirate":
        return "Respond as a pirate."
    elif context.style == "robot":
        return "Respond as a robot and say 'beep boop' a lot."


agent = Agent[CustomContext](
    name="Dynamic Prompt Agent",
    instructions=custom_instructions,
)

async def async_main():
    context = CustomContext(style="robot")
    with trace("Dynamic Prompt Example"):
        result = await Runner.run(
            agent,
            input="Tell me a joke",
            run_config=config,
            context=context
        )
        print(result.final_output)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()














