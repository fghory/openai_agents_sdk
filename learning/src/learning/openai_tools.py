
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, WebSearchTool # type: ignore
from openai.types.responses import ResponseTextDeltaEvent # type: ignore
from agents.run import RunConfig # type: ignore
from dotenv import load_dotenv
import asyncio


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY),
)

config = RunConfig(workflow_name="Web Search")

async def async_main() -> None:
    agent = Agent(
        name="Search Agent",
        instructions="You are a helpful assistant. You are given a question and you need to search the web for the answer.",
        tools=[WebSearchTool()],
        model=model
    )

    result = Runner.run_streamed(agent, input="How good are JF-17 Thunder Block 3 aircraft?", run_config=config)
    async for event in result.stream_events():
        if event.type=="raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

def main():
    """Entry point for the package when called as a script."""
    import asyncio
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
