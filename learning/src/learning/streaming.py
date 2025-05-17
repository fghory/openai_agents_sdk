
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, enable_verbose_stdout_logging
from openai.types.responses import ResponseTextDeltaEvent
from agents.run import RunConfig
from dotenv import load_dotenv
import asyncio


load_dotenv()
enable_verbose_stdout_logging()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",
    openai_client=AsyncOpenAI(api_key=OPENAI_API_KEY),
)

config = RunConfig(workflow_name="Joker")

async def async_main() -> None:
    agent = Agent(
        name="Joker",
        instructions="You are a helpful assistant.",
    )

    result = Runner.run_streamed(agent, input="Please tell me 5 jokes", run_config=config)
    
    async for event in result.stream_events():
        print(event)
        # if event.type=="raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
        #     print(event.data.delta, end="", flush=True)
           

def main():
    """Entry point for the package when called as a script."""
    import asyncio
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
