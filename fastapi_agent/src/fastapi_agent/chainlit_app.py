import chainlit as cl
from panaversity import model, panacloud_agent
from agents import Runner, RunConfig
from openai.types.responses import ResponseTextDeltaEvent



config = RunConfig(model=model, tracing_disabled=True)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to the Panacloud Chatbot!").send()

@cl.on_message
async def on_message(message: cl.Message):
    # Create an empty message to stream content into
    msg = cl.Message(content="")

    # Start streaming the agent's response
    result = Runner.run_streamed(panacloud_agent, input=message.content, run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            token = event.data.delta or ""
            await msg.stream_token(token)

    # Finalize the message after streaming is complete
    await msg.update()

