from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents.llm import function_tool
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import (
    openai,
    tavus,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.turn_detector.english import EnglishModel

from langchain_openai import AzureOpenAIEmbeddings
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import Settings
Settings.llm = None

load_dotenv()

embedding = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_CODE_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_CODE_ENDPOINT"),
    azure_deployment="text-embedding-3-large", 
    api_version="2024-12-01-preview",
    chunk_size=2000
)

client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("pdfs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embedding,
)
db = index.as_query_engine(llm=None)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful HR assistant. Use vectorstore for answering the HR related questions or anything you're not sure of")

    @function_tool
    async def query_vectorstore(self, context: RunContext, question: str):
        """Return the relevant HR docs to the query from vectostore"""
        docs = db.query(question)
        print(f"docs: {docs}")
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]



async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        #stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.realtime.RealtimeModel.with_azure(
            azure_deployment="gpt-4o-mini-realtime-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # or AZURE_OPENAI_ENDPOINT
            api_key=os.getenv("AZURE_OPENAI_API_KEY"), # or AZURE_OPENAI_API_KEY
            api_version="2024-10-01-preview", # or OPENAI_API_VERSION
        ),
        #tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=EnglishModel(),
    )
    
    avatar = tavus.AvatarSession(
        replica_id="r4c41453d2",
        persona_id="p2fbd605"
    )

    await avatar.start(session, room=ctx.room)
    

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))