from agents import Agent, Runner, OpenAIChatCompletionsModel, RunConfig, ItemHelpers, MessageOutputItem, trace, TResponseInputItem
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

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


spanish_agent = Agent(
    name="Spanish Translator",
    instructions="""You are a Spanish translator. You translate the user's input from English to Spanish.""",
)

translation_picker = Agent(
    name="Translation Picker",
    instructions="""You are a translation picker. You pick the best translation from the list of translations.""",
)


async def async_main()->None:
    msg = input("What do you want to translate to Spanish?: ")
    trans1, trans2, trans3 = await asyncio.gather(
        Runner.run(spanish_agent, msg, run_config=config),
        Runner.run(spanish_agent, msg, run_config=config),
        Runner.run(spanish_agent, msg, run_config=config),
    )

    outputs = [
        ItemHelpers.text_message_outputs(trans1.new_items),
        ItemHelpers.text_message_outputs(trans2.new_items),
        ItemHelpers.text_message_outputs(trans3.new_items),
    ]

    translations = "\n\n".join(outputs)
    print(f"\n\n    Translations:\n\n{translations}")

    pick = await Runner.run(
        translation_picker,
        translations,
        run_config=config
    )

    print(f"\n\n    Pick:\n\n{pick.final_output}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()









