from dotenv import load_dotenv
import asyncio
from langchain_tavily import TavilySearch  # type: ignore
from tavily import TavilyClient
import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, WebSearchTool, function_tool # type: ignore 
from openai.types.responses import ResponseTextDeltaEvent # type: ignore
from agents.run import RunConfig # type: ignore

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
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
def tavily_search(query: str) -> str:
    results = tavily_client.search(query, max_results=3)
    return results['results']

agent = Agent(
    name="Tavily Search Agent",
    instructions="""You are a helpful assistant that can search the web for information.
    You are given a query and you need to search the web for information.
    You return the most relevant information from the web along with the urls.
    """,
    tools=[tavily_search],
)
 
def main():
    result = Runner.run_sync(agent, input="What is China's latest 6th generation fighter jet?", run_config=config)
    print(result.final_output)

if __name__ == "__main__":
    main()

