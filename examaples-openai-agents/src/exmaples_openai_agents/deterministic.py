from agents import Agent, Runner, ItemHelpers, MessageOutputItem, RunConfig, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
    tracing_disabled=True,
)


outline_agent = Agent(
    name="Outline Agent",
    instructions="""Generate a very short story outline based on the user's input."""
    )

class OutlineChecker(BaseModel):
    good_quality: bool
    is_scifi: bool

outline_checker_agent = Agent(
    name="Outline Checker",
    instructions="""Read the given story outline, and judge the quality. Also, determine if it is a scifi story.""",
    output_type=OutlineChecker,
    )

story_writer_agent = Agent(
    name="Story Writer",
    instructions="""Write a short story based on the given outline.""",
    output_type=str,
    )

async def async_main():
    outline = await Runner.run(outline_agent, "A story about a robot that can fly.", run_config=config)
    print(outline)
    checker = await Runner.run(outline_checker_agent, outline.final_output, run_config=config)
    
    assert isinstance(checker.final_output, OutlineChecker)

    if not checker.final_output.good_quality:
        print("Outline is not good quality, we stop here")
        exit(0)

    if not checker.final_output.is_scifi:
        print("Outline is not scifi, we stop here")
        exit(0)

    story = await Runner.run(story_writer_agent, outline.final_output, run_config=config)
    print(story.final_output)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()





