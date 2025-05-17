from agents import Agent, FileSearchTool, Runner, enable_verbose_stdout_logging, set_default_openai_key
import asyncio, os
from dotenv import load_dotenv


enable_verbose_stdout_logging()
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_default_openai_key(str(OPENAI_API_KEY))

agent = Agent(
    name="Assistant",
    tools=[
        
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["vs_681a24009c508191a4e2523c6c746c9c","vs_681a32320de881918291bf7188c7e7fc"],
            include_search_results=True,
            ranking_options= {
                "ranker":"default-2024-11-15",
                "score_threshold":0.1
            }
        ),
    ],
)

async def async_main():
    result = await Runner.run(agent, "How many Masters degrees does Fahad have and What does attention means in Attention all you need paper?")
    print(result.final_output)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()    