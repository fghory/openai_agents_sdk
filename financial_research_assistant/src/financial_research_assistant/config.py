import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel # type: ignore 
from agents.run import RunConfig # type: ignore
from dotenv import load_dotenv



load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",   
)

model_flash = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

model_pro = OpenAIChatCompletionsModel(
    model="gemini-2.5-pro-exp-03-25",
    openai_client=external_client
)


config_flash= RunConfig(
    model=model_flash,
    model_provider=external_client,
    tracing_disabled=True,
)

config_pro = RunConfig(
    model=model_pro,
    model_provider=external_client,
    tracing_disabled=True,
)
