from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig    
from dotenv import load_dotenv
import os
from agents import set_default_openai_key



load_dotenv()
# set_default_openai_key(os.getenv("OPENAI_API_KEY"))
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


def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = Runner.run_sync(agent, "Write a haiku about recursion in programming.", run_config=config)
    print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.
