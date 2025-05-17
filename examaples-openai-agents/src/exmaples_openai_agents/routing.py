from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, ItemHelpers, MessageOutputItem, trace, TResponseInputItem
from pydantic import BaseModel
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import os
import asyncio
from typing import Literal
import agentops


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


french_agent = Agent(
    name="French Agent",
    instructions="""You are a French agent. You translate the user's input from English to French.""",
    handoff_description="""This agent translates the user's input from English to French.""",
)

italian_agent = Agent(
    name="Italian Agent",
    instructions="""You are an Italian agent. You translate the user's input from English to Italian.""",
    handoff_description="""This agent translates the user's input from English to Italian.""",
)

routing_agent = Agent(
    name="Routing Agent",
    instructions="""You are a routing agent. You route the user's input to the appropriate language agent.""",
    handoffs=[french_agent, italian_agent],
)

async def async_main()->None:
    msg = input("What do you want to translate to French or Italian?: ")
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    with trace("Routing example"):
        route = await Runner.run(
            routing_agent,
            input=inputs,
            run_config=config
        )
        print(f"\n\n    Route:\n\n{route.final_output}")
        # async for event in route.stream_events():
        #     if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        #         print(event.data.delta, end="", flush=True)

def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()












