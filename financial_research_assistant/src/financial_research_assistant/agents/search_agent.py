from agents import Agent, WebSearchTool, function_tool
from agents.model_settings import ModelSettings
from tavily import TavilyClient # type: ignore
from dotenv import load_dotenv
import os

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

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

# Given a search term, use web search to pull back a brief summary.
# Summaries should be concise but capture the main financial points.
INSTRUCTIONS = (
    "You are a research assistant specializing in financial topics. "
    "Given a search term, use web search to retrieve up‑to‑date context and "
    "produce a short summary of at most 300 words. Focus on key numbers, events, "
    "or quotes that will be useful to a financial analyst."
)

search_agent = Agent(
    name="FinancialSearchAgent",
    instructions=INSTRUCTIONS,
    tools=[tavily_search_tool],
)