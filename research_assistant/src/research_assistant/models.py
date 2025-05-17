from agents import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
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
    model="gemini-2.5-pro-preview-05-06",
    openai_client=external_client
)