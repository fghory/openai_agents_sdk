import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, WebSearchTool, function_tool # type: ignore 
from openai.types.responses import ResponseTextDeltaEvent # type: ignore
from agents.run import RunConfig # type: ignore
from dotenv import load_dotenv
import asyncio
from langchain_tavily import TavilySearch  # type: ignore
from tavily import TavilyClient


load_dotenv()


# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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

tavily_tool = TavilySearch(max_results=3)

# Define a wrapper function for the OpenAI Agents SDK
@function_tool
def tavily_search_tool(query: str) -> str:
    """Search the web using Tavily Search."""
    try:
        results = tavily_client.search(query, max_results=3)
        # Format results as a string or JSON for the agent
        formatted_results = results['results']
        return formatted_results
    except Exception as e:
        return f"Error during search: {str(e)}"


async def async_main() -> None:
    agent = Agent(
        name="Search Agent",
        instructions="You are a helpful assistant. You are given a question and you need to search the web for the answer.",
        tools=[tavily_search_tool],
    )

    result = Runner.run_streamed(agent, input="Are we heading for a recession?", run_config=config)
    async for event in result.stream_events():
        if event.type=="raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

def main():
    """Entry point for the package when called as a script."""
    import asyncio
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
