from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool
from dotenv import load_dotenv
import os

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

@function_tool
def get_weather(destination: str):
    """Get the weather of the destination"""
    return f"The weather in {destination} is sunny with a temperature of 20 degrees Celsius."

agent = Agent(
    name="Weather Agent",
    instructions="You are a weather agent that can get the weather of a destination.",
    tools=[get_weather],
)

def main():
    queries = [
        "What is the weather in Rotterdam?",
        "What is the weather in Paris?",
    ]   
    
    for query in queries:
        print(f"Query: {query}")
        result = Runner.run_sync(agent, input=query, run_config=config)
        print(result.final_output)
        print("-"*100)

if __name__ == "__main__":
    main()
