import json, random, os, asyncio
from agents import Agent, HandoffInputData, Runner, function_tool, handoff, trace, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from agents.extensions import handoff_filters
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",   
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)
                           

@function_tool
def random_number_tool(max:int)->int:
    """Return a random integer between 0 and the given maximum."""
    return random.randint(0, max)

def spanish_handoff_message_filter(handoff_message_data:HandoffInputData)->HandoffInputData:
    print(f"pre_handoff_items: {handoff_message_data.pre_handoff_items}")
    print(f"new_items: {handoff_message_data.new_items}")
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # Second, we'll also remove the first two items from the history, just for demonstration
    history = (
        tuple(handoff_message_data.input_history[2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )


first_agent = Agent(
    name="Assistant",
    instructions="Be extremely concise.",
    tools=[random_number_tool],
)

spanish_agent = Agent(
    name="Spanish Assistant",
    instructions="You only speak Spanish and are extremely concise.",
    handoff_description="A Spanish-speaking assistant.",
)

second_agent = Agent(
    name="Assistant",
    instructions=(
        "Be a helpful assistant. If the user speaks Spanish, handoff to the Spanish assistant."
    ),
    handoffs=[handoff(spanish_agent, input_filter=spanish_handoff_message_filter)],
    tools=[random_number_tool],
)


async def async_main():
    # Step 1: Initial conversation
    result = await Runner.run(first_agent, "Hi, my name is Sora", run_config=config)
    print("Step 1 done")
    
    # Step 2: Random number request
    messages = result.to_input_list()
    messages.append({"role": "user", "content": "Can you generate a random number between 0 and 100?"})
    result = await Runner.run(second_agent, messages, run_config=config)
    print("Step 2 done")

    # Step 3: Population question
    messages = result.to_input_list()
    messages.append({"role": "user", "content": "I live in New York City. Whats the population of the city?"})
    result = await Runner.run(second_agent, messages, run_config=config)
    print("Step 3 done")

    # Step 4: Spanish handoff
    messages = result.to_input_list()
    messages.append({"role": "user", "content": "Por favor habla en español. ¿Cuál es mi nombre y dónde vivo?"})
    result = await Runner.run(second_agent, messages, run_config=config)
    print("Step 4 done")

    print("\n===Final messages===\n")
    for message in result.to_input_list():
        print(json.dumps(message, indent=2))

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()