from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, UTC
from uuid import uuid4
from panaversity import model, panacloud_agent
# Import OpenAI Agents SDK
from agents import Agent, Runner, function_tool, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from fastapi.responses import StreamingResponse
import json
from chainlit.utils import mount_chainlit


config = RunConfig(model=model, tracing_disabled=True)

# Initialize the FastAPI app
app = FastAPI(
    title="Panacloud Chatbot API",
    description="A FastAPI-based API for Panacloud Chatbot",
    version="0.1.0",
)

# Mount the Chainlit app at the "/chainlit" path
mount_chainlit(app=app, target="chainlit_app.py", path="/chainlit")

# Pydantic models
class Metadata(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    session_id: str = Field(default_factory=lambda: str(uuid4()))

class Message(BaseModel):
    user_id: str
    text: str
    metadata: Metadata | None = None
    tags: list[str] | None = None

class Response(BaseModel):
    user_id: str
    reply: str
    metadata: Metadata


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Panacloud Chatbot API!"}

# GET endpoint with query parameters
@app.get("/users/{user_id}")
async def get_user(user_id: str, role: str | None = None):
    user_info = {"user_id": user_id, "role": role if role else "guest"}
    return user_info

# POST endpoint for chatting
@app.post("/chat/", response_model=Response)
async def chat(message: Message):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text cannot be empty")

    # Use the OpenAI Agents SDK to process the message
    result = await Runner.run(panacloud_agent, input=message.text, run_config=config)
    reply_text = result.final_output  # Get the agent's response

    return Response(
        user_id=message.user_id,
        reply=reply_text,
        metadata=Metadata()
    )

# POST endpoint for chatting
async def stream_response(message: Message):
    result = Runner.run_streamed(panacloud_agent, input=message.text, run_config=config)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
            # Serialize dictionary to JSON string
            chunk = json.dumps({"chunk": event.data.delta})
            yield f"data: {chunk}\n\n"
            
@app.post("/chat/stream", response_model=Response)
async def chat_stream(message: Message):
    if not message.text.strip():
        raise HTTPException(
            status_code=400, detail="Message text cannot be empty")

    return StreamingResponse(
        stream_response(message),
        media_type="text/event-stream"
    )
