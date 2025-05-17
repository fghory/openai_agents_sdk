from agents import (
    Agent, InputGuardrail, GuardrailFunctionOutput, Runner, OpenAIChatCompletionsModel,
    RunConfig, AsyncOpenAI, RunContextWrapper,
    TResponseInputItem, input_guardrail
)
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
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


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="CHeck if the user is asking about homework.",
    output_type=HomeworkOutput
    )

math_tutor_agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
    handoff_description="Specialist agent for math questions.",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

@input_guardrail
async def homework_guardrail(ctx:RunContextWrapper[None], agent:Agent, input:str|list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered= not result.final_output.is_homework,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[homework_guardrail],
)


async def async_main() -> None:
    # result = await Runner.run(triage_agent, "who was the first president of the united states?", run_config=config)
    # print(result.final_output)

    result = await Runner.run(triage_agent, "I need to solve what is 25 + 3 as my homework", run_config=config)
    print(result.final_output)

def main():
    """Entry point for the package when called as a script."""
    import asyncio
    asyncio.run(async_main())
    
if __name__ == "__main__":
    main()

    










