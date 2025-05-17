from agents import Agent, Runner, enable_verbose_stdout_logging, set_default_openai_key, WebSearchTool
import asyncio, os
from dotenv import load_dotenv


enable_verbose_stdout_logging()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_default_openai_key(str(OPENAI_API_KEY))

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(search_context_size="low")
    ],
)

async def async_main():
    result = await Runner.run(agent, "Why there is tension between Pakistan and India these days?")
    print(result.final_output)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()    