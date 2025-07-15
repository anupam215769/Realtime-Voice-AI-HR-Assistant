# HR Assistant with Vector Search and LiveKit

This project is an AI-powered HR assistant that leverages modern LLMs, vector search, and real-time communication features. It uses LangChain, LlamaIndex, ChromaDB, and LiveKit Agents to provide intelligent document retrieval and conversational HR support via voice/video.

---

## Features

- Ingest and chunk PDF documents for HR knowledge
- Generate and store vector embeddings using Azure OpenAI
- Semantic search over HR documents via vector store (ChromaDB)
- Real-time, voice-enabled HR assistant powered by LiveKit Agents and OpenAI LLMs
- Extensible plugin architecture for speech-to-text, TTS, VAD, and avatars

---

## Tech Stack

- Python
- LangChain & LlamaIndex
- ChromaDB
- Azure OpenAI (Embeddings & Chat)
- LiveKit Agents & Plugins

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/hr-assistant-livekit.git
cd hr-assistant-livekit
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env` and fill in the required keys:

```env
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
LLAMA_PARSE_KEY=your_llama_parse_key
TAVUS_API_KEY=your_tavus_api_key
LIVEKIT_URL=wss://your.livekit.server
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```

### 4. Prepare Your HR Documents

Place your HR-related PDFs in the `./pdfs` directory.

### 5. LiveKit React Frotend
```bash
git clone https://github.com/livekit-examples/agent-starter-react.git
```

```bash
cd agent-starter-react
npm install
```

create .env.local inside and fill in the required keys
```env
LIVEKIT_URL=wss://your.livekit.server
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```


## Usage

### 1. Index the Documents (Need to run only one time for creating db)

This will parse, chunk, embed, and store your PDFs for later querying:

```bash
python db_creation.py
```

### 2. Start the LiveKit Agent

```bash
python agents.py download-files
```

```bash
python agents.py dev
```

### 3. Start Frontend

```bash
npm run dev
```
And open [http://localhost:3000](http://localhost:3000) in your browser.
