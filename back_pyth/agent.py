import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from livekit import agents
from livekit.agents import AgentSession, RoomInputOptions
from livekit.plugins import google, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

app = FastAPI()

# Modèle pour recevoir le body JSON
class JoinRoomRequest(BaseModel):
    room_name: str

class Assistant(agents.Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a psychological voice assistant. Your job is to warmly greet the user..."
        )

@app.post("/join-room")
async def join_room(data: JoinRoomRequest):
    """Lance l'assistant et le connecte à la room LiveKit"""

    async def run_agent():
        session = AgentSession(
            livekit_url=os.getenv("LIVEKIT_URL"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
            stt=deepgram.STT(model="nova-3", language="multi"),
            llm=google.LLM(model="gemini-2.0-flash-exp", temperature=0.8),
            tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
        )

        await session.start(
            room=data.room_name,
            agent=Assistant(),
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )

        await session.say(
            "Hello. I am your psychological voice assistant. How are you feeling today?"
        )

        await session.send_chat_message(
            "Hello. I am your psychological voice assistant. How are you feeling today?"
        )

    # On lance l'agent en tâche de fond
    asyncio.create_task(run_agent())

    # Réponse immédiate pour ne pas bloquer le front
    return {"message": f"Assistant is joining room {data.room_name}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
