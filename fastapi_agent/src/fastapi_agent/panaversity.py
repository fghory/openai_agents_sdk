from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, function_tool, enable_verbose_stdout_logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
from pydantic import BaseModel
from tavily import TavilyClient
import agentops
from agents.extensions.visualization import draw_graph


enable_verbose_stdout_logging()
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")

agentops.init(AGENTOPS_API_KEY)

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini",  # Replace with your desired model
    openai_client=openai_client
)

config = RunConfig(model=model)

# external_client = AsyncOpenAI(
#     api_key=GEMINI_API_KEY,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",   
# )

# model = OpenAIChatCompletionsModel(
#     model="gemini-1.5-flash",
#     openai_client=external_client
# )

# config = RunConfig(
#     model=model,
#     model_provider=external_client,
#     # tracing_disabled=True,
# )

@function_tool
def tavily_search(query: str) -> str:
    results = tavily_client.search(query, max_results=3)
    return results['results']


webdev_agent = Agent(
    name="Web Development Agent",
    instructions="You are a seasoned Web developer. You answer queries about web development.",
    handoff_description="This agent answers queries about web development."
)

mobiledev_agent = Agent(
    name="Mobile Development Agent",
    instructions="You are a seasoned mobile application developer. You answer queries about mobile application development.",
    handoff_description="This agent answers queries about mobile application development."
)

devops_agent = Agent(
    name="Dev Ops Agent",
    instructions="You are a seasoned Dev Ops Engineer. You answer queries about Dev Ops.",
    handoff_description="This agent answers queries about dev ops."
)

openai_agent = Agent(
    name="OpenAI SDK Agent",
    instructions="""You have exclusive knowledge about OpenAI Agents SDK.
      You always use the given tool to search web for queries about OpenAI Agents SDK.
      You return the most relevant information from the web along with the urls.
      """,
    tools=[tavily_search],  
    handoff_description="This agent answers queries related to OpenAI Agents SDK."
)


agentic_ai_agent = Agent(
    name="Agentic AI Agent",
    instructions="You are a Agentic AI agent. You always use respective agent to answer user query.",
    handoff_description="This agent uses respective tools to answer user queries about Dev Ops and OpenAI Agents SDK.",
    tools=[devops_agent.as_tool
        (
        tool_name="transfer_to_devops",
        tool_description="Answer queries about Dev Ops",
    ),
    openai_agent.as_tool
        (
        tool_name="transfer_to_openai",
        tool_description="answers queries related to OpenAI Agents SDK",
    ),
    ]
    
)

panacloud_agent = Agent(
    name="Panacloud Agent",
    instructions="You are a triage agent. You transfer to respective agent to answer user query.",
    handoffs=[webdev_agent, mobiledev_agent, agentic_ai_agent]
)

async def async_main():
    
    result = await Runner.run(panacloud_agent,"Tell me about OpenAI Agents SDK", run_config=config)
    print(result.final_output)
    print(result.last_agent.name)
    draw_graph(panacloud_agent)

def main():
    asyncio.run(async_main())
    draw_graph(panacloud_agent, filename="panacloud_graph")

if __name__ == "__main__":
    main()

