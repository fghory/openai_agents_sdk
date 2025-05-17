from agents import (
    Agent, Runner, OpenAIChatCompletionsModel, RunConfig, ItemHelpers,
    MessageOutputItem, trace, enable_verbose_stdout_logging
    )
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio

"""
This example shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools. In this case, it picks from a set of translation
agents.
"""

enable_verbose_stdout_logging()
load_dotenv()
Gemini_API_KEY = os.getenv("GEMINI_API_KEY")
OpenAI_API_KEY = os.getenv("OPENAI_API_KEY")

# external_client = AsyncOpenAI(
#     api_key=Gemini_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",   
# )   

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client
# )   

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     tracing_disabled=True,
# )


spanish_agent = Agent(
    name="Spanish Translator",
    instructions="""You are a helpful assistant that translates English to Spanish.""",
    handoff_description="An english to spanish translator",
)

french_agent = Agent(
    name="French Translator",
    instructions="""You are a helpful assistant that translates English to French.""",
    handoff_description="An english to french translator",
)

italian_agent = Agent(
    name="Italian Translator",
    instructions="""You are a helpful assistant that translates English to Italian.""",
    handoff_description="An english to italian translator",
)

orchestrator_agent = Agent(
    name="Orchestrator",
    instructions="""
        You are a translation agent. You use the tools given to you to translate.
        If asked for multiple translations, you call the relevant tools in order.
        You never translate on your own, you always use the provided tools.
""",
    tools=[
        spanish_agent.as_tool(
        tool_name="translate_to_spanish",
        tool_description="Translate the user's message to Spanish",
    ),
    french_agent.as_tool(
        tool_name="translate_to_french",
        tool_description="Translate the user's message to French",
    ),
    italian_agent.as_tool(
        tool_name="translate_to_italian",
        tool_description="Translate the user's message to Italian",
    ),
    ],
)


synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You inspect translations, correct them if needed, and produce a final concatenated response.",
)

async def async_main():
    msg = input("Hi, what would you like translated and in which language?")
    orchestrator_result = await Runner.run(orchestrator_agent,msg)
    print(orchestrator_result)

    for item in orchestrator_result.new_items:
        if isinstance(item, MessageOutputItem):
            text= ItemHelpers.text_message_output(item)
            if text:
                print(f"    - Translation Step: {text}")

    synthesizer_result = await Runner.run(
            synthesizer_agent, orchestrator_result.to_input_list()
        )

    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")

def main():
    """Entry point for the package when called as a script."""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
