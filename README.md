# Local RAG Tutorial with Docker Compose

This is a starter tutorial for Retrieval-Augmented Generation (RAG) running 100% locally on your machine over Docker Compose.

## Architecture

1. **Ollama**: Serves the LLM (`llama3`) and the Embedding model (`nomic-embed-text`).
2. **Qdrant**: Stores our document vectors.
3. **App**: A Python environment using LangChain to connect to Ollama and Qdrant.

## Prerequisites
Before you begin, ensure you have the following installed on your machine:
* **Docker**: The core container runtime. [Installation guide](https://docs.docker.com/engine/install/)
* **Docker Compose**: The tool to run multi-container setups. 
  * If your system uses `docker-compose` (with a hyphen), you can usually install it via your package manager (e.g., `sudo apt install docker-compose`).
  * If you prefer the modern `docker compose` plugin, follow the [official plugin installation guide](https://docs.docker.com/compose/install/linux/).


## Getting Started

First, navigate to the tutorial directory:
```bash
cd /home/yevhenvasko/projects/ai-rag-tutorial
```

### 1. Start the services
Run this to spin up the entire cluster:
```bash
docker compose up -d
```
Docker will build the Python environment (installing Langchain and Qdrant client) and start Ollama and Qdrant in the background.

### 2. Download the models into Ollama
Ollama starts empty. You need to pull the models you intend to use. Run these two commands in your terminal:
```bash
docker exec -it ollama ollama pull llama3
docker exec -it ollama ollama pull nomic-embed-text
```

### 3. Run the application
You can interact with your RAG system in two ways.

**Option A: Command Line Script**
The simple `app/main.py` is ready to execute:
```bash
docker exec -it rag_app python main.py
```

**Option B: Web UI (Streamlit)**
A full Web UI has been built for you to upload text and chat with the AI! 
Run this to start the web server in the background:
```bash
docker exec -d rag_app streamlit run ui.py
```
*Once started, open your browser and go to [http://localhost:8501](http://localhost:8501) to use it!*

## How to explore
1. Edit `app/main.py` directly from your host machine (it is volume-mounted to the container).
2. Just rerun `docker exec -it rag_app python main.py` to test your changes. 
3. You can add PDFs, text files, or any other data into the `app/` folder, install standard `PyPDF2` or `unstructured` tools via `app/requirements.txt`, and experiment with larger document ingestions!

## Stop and clean up
When you're done:
```bash
docker compose down
```
If you want to clear your vector database and model downloads completely:
```bash
docker compose down -v
```
