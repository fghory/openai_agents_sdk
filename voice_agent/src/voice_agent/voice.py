import sounddevice as sd
from dotenv import load_dotenv
from agents import set_default_openai_key, Agent
from agents.voice import SingleAgentVoiceWorkflow, TTSModelSettings, VoicePipelineConfig, VoicePipeline, AudioInput
import os
import asyncio
import numpy as np
from typing import List, Optional, Any
from numpy.typing import NDArray

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
set_default_openai_key(api_key)

def get_audio_devices() -> tuple[dict[str, Any], dict[str, Any], float, float]:
    input_device = sd.query_devices(kind='input')
    output_device = sd.query_devices(kind='output')
    in_samplerate = input_device['default_samplerate']
    out_samplerate = output_device['default_samplerate']
    return input_device, output_device, in_samplerate, out_samplerate

def record_audio(in_samplerate: float) -> NDArray[np.int16]:
    recorded_chunks: List[NDArray[np.int16]] = []
    
    with sd.InputStream(
        samplerate=in_samplerate,
        channels=1,
        dtype='int16',
        callback=lambda indata, frames, time, status: recorded_chunks.append(indata.copy())
    ):
        input("Press Enter to stop recording...")
    
    return np.concatenate(recorded_chunks)

def create_agent() -> Agent:
    return Agent(
        name="Assistant",
        instructions=(
            "Repeat the user's question back to them, and then answer it. Note that the user is "
            "speaking to you via a voice interface, although you are reading and writing text to "
            "respond. Nonetheless, ensure that your written response is easily translatable to voice."
        ),
        model="gpt-4.1-nano"
    )

def create_pipeline(agent: Agent) -> VoicePipeline:
    workflow = SingleAgentVoiceWorkflow(agent)
    custom_tts_settings = TTSModelSettings(
        instructions=(
            "Personality: upbeat, friendly, persuasive guide.\n"
            "Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making "
            "the listener feel confident and comfortable.\n"
            "Pronunciation: Clear, articulate, and steady, ensuring each instruction is "
            "easily understood while maintaining a natural, conversational flow.\n"
            "Tempo: Speak relatively fast, include brief pauses and after before questions.\n"
            "Emotion: Warm and supportive, conveying empathy and care, ensuring the listener "
            "feels guided and safe throughout the journey."
        )
    )
    voice_pipeline_config = VoicePipelineConfig(tts_settings=custom_tts_settings)
    return VoicePipeline(workflow=workflow, config=voice_pipeline_config)

async def process_audio(audio_input: AudioInput, pipeline: VoicePipeline) -> NDArray[np.int16]:
    result = await pipeline.run(audio_input=audio_input)
    response_chunks: List[NDArray[np.int16]] = []
    
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            response_chunks.append(event.data)
    
    return np.concatenate(response_chunks, axis=0)

async def voice_assistant_optimized() -> None:
    input_device, output_device, in_samplerate, out_samplerate = get_audio_devices()
    agent = create_agent()
    pipeline = create_pipeline(agent)
    openai_sample_rate = 24_000

    while True:
        cmd = input("Press Enter to speak (or type 'q' to exit): ")
        if cmd.lower() == "q":
            print("Exiting...")
            break
            
        print("Listening...")
        recording = record_audio(in_samplerate)
        audio_input = AudioInput(buffer=recording)
        
        print("Assistant is responding...")
        response_audio_buffer = await process_audio(audio_input, pipeline)
        
        sd.play(response_audio_buffer, samplerate=openai_sample_rate)
        sd.wait()
        print("---")

async def main() -> None:
    await voice_assistant_optimized()

if __name__ == "__main__":
    asyncio.run(main())
