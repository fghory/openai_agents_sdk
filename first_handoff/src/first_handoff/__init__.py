from agents import Agent, Runner
from dotenv import load_dotenv
import os
from agents import set_default_openai_key

load_dotenv()
set_default_openai_key(os.getenv("OPENAI_API_KEY"))


history_tutor_agent = Agent(
name="History Tutor",
handoff_description="Specialist agent for historical questions",
instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
name="Math Tutor",
handoff_description="Specialist agent for math questions",
instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

triage_agent = Agent(
name="Triage Agent",
instructions="You determine which agent to use based on the user's homework question",
handoffs=[history_tutor_agent, math_tutor_agent]
)


async def async_main() -> None:
    result = await Runner.run(triage_agent, "How do you calculate the area of a circle?")
    print(result.final_output)
    return result.final_output

def main():
    """Entry point for the package when called as a script."""
    import asyncio
    asyncio.run(async_main())
    
if __name__ == "__main__":
    main()
