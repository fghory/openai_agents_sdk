# main.py (simplified)
import asyncio, os
import sounddevice as sd #type:ignore
from agents.voice import StreamedAudioInput, VoicePipeline
from my_workflow import MyWorkflow #type:ignore
from agents import set_default_openai_key
SAMPLE_RATE = 24000
CHANNELS = 1
from dotenv import load_dotenv


load_dotenv()
set_default_openai_key(os.getenv("OPENAI_API_KEY"))

async def record_and_send(audio_input, event):
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS)
    stream.start()
    try:
        while True:
            await event.wait()
            data, _ = stream.read(int(SAMPLE_RATE * 0.02))
            await audio_input.add_audio(data)
    finally:
        stream.close()

async def run_pipeline(secret):
    audio_input = StreamedAudioInput()
    workflow = MyWorkflow(secret, lambda t: print(f"User said: {t}"))
    pipeline = VoicePipeline(workflow=workflow)

    result = await pipeline.run(audio_input)
    async for evt in result.stream():
        if evt.type == "voice_stream_event_audio":
            sd.play(evt.data, SAMPLE_RATE)
        else:
            print(evt.event)

async def main():
    send_event = asyncio.Event()
    send_event.set()
    audio_input = StreamedAudioInput()

    await asyncio.gather(
        run_pipeline("dog"),
        record_and_send(audio_input, send_event),
    )

if __name__ == "__main__":
    asyncio.run(main())
