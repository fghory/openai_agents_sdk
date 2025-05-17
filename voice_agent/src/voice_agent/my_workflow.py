# my_workflow.py (simplified)
import random
from agents import Agent, Runner, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import VoiceWorkflowBase, VoiceWorkflowHelper
from collections.abc import AsyncIterator

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is {random.choice(['sunny','cloudy','rainy','snowy'])}."

spanish_agent = Agent(
    name="Spanish",
    instructions=prompt_with_handoff_instructions("Speak politely in Spanish."),
    model="gpt-4o-mini",
)

assistant_agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "Be polite. If user speaks Spanish, hand off to Spanish agent."
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)

class MyWorkflow(VoiceWorkflowBase):
    def __init__(self, secret_word: str, on_start):
        self.history = []
        self.agent = assistant_agent
        self.secret = secret_word.lower()
        self.on_start = on_start

    async def run(self, transcription: str) -> AsyncIterator[str]:
        # user transcription callback
        self.on_start(transcription)
        self.history.append({"role": "user", "content": transcription})

        # secret‑word shortcut
        if self.secret in transcription.lower():
            yield "You guessed the secret word!"
            return

        # invoke agent and stream response
        result = Runner.run_streamed(self.agent, self.history)
        async for chunk in VoiceWorkflowHelper.stream_text_from(result):
            yield chunk

        # update for next turn
        self.history = result.to_input_list()
        self.agent = result.last_agent
