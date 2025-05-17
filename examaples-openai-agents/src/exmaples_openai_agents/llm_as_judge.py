from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, RunConfig, ItemHelpers, MessageOutputItem, trace,
    TResponseInputItem,
)
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from typing import Literal

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

story_outline_generator = Agent(
    name="Story Outline Generator",
    instructions="""You generate a very short story outline based on the user's input.
        If there is any feedback provided, use it to improve the outline.""",
)

class EvaluationFeedback(BaseModel):
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions=""""You evaluate a story outline and decide if it's good enough."
        "If it's not good enough, you provide feedback on what needs to be improved."
        "Never give it a pass on the first try.""",
    output_type=EvaluationFeedback,
)

async def async_main()->None:
    msg = input("What story sould you like me to write: ")
    input_items: list[TResponseInputItem] = [{"content": msg, "role": "user"}]
    latest_outline: str|None = None
    while True:
        outline = await Runner.run(story_outline_generator, input_items, run_config=config)
        input_items = outline.to_input_list()
        latest_outline = ItemHelpers.text_message_outputs(outline.new_items)
        eval = await Runner.run(evaluation_agent, input_items, run_config=config)
        result: EvaluationFeedback = eval.final_output
        print(f"Evaluation: {result.score}")

        if result.score == "pass":
            print(f"Passed! Story line is good enough.")
            break
        else:
            print(f"Failed! Here's the feedback: {result.feedback}")

        input_items.append({"content": result.feedback, "role": "user"})

    print(f"Final story outline: {latest_outline}")


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()



