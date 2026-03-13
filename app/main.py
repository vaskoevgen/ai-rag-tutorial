import time
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate

# URLs for services inside the Docker Network
OLLAMA_URL = "http://ollama:11434"
QDRANT_URL = "http://qdrant:6333"

def main():
    print("🚀 Starting Local RAG Example...")

    print("Initializing embedding model (nomic-embed-text) and LLM (llama3)...")
    # Initialize the tools to communicate with Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL,
    )
    
    llm = ChatOllama(
        model="llama3",
        base_url=OLLAMA_URL,
        temperature=0.2, # Lower temperature gives more factual, precise answers
    )

    print("\n--- Phase 1: Ingestion ---")
    sample_knowledge = [
        "The secret keyword to unlock the RAG system is: 'PINEAPPLE'.",
        "Docker Compose orchestrates multiple containers like Ollama, Qdrant, and the Python app.",
        "Retrieval-Augmented Generation (RAG) combines search with LLMs to reduce AI hallucinations.",
        "Qdrant is a fast vector database written in Rust that we use to store vector embeddings."
    ]
    
    # Convert text strings into LangChain "Document" objects
    docs = [Document(page_content=text) for text in sample_knowledge]
    
    print("Saving documents into Qdrant vector database...")
    # This automatically connects to Qdrant and saves the embedded vectors
    vector_store = QdrantVectorStore.from_documents(
        docs,
        embeddings,
        url=QDRANT_URL,
        collection_name="learning_rag",
        force_recreate=True,  # Overwrites history to start fresh every time we run script
    )
    print("✅ Ingestion complete.")

    print("\n--- Phase 2: Retrieval ---")
    query = "What is the secret keyword for the RAG system?"
    print(f"User Query: '{query}'")
    
    # We ask the Vector DB to find the 'k' most similar pieces of context
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    print("\nFound relevant context from Qdrant:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  [{i+1}] {doc.page_content}")
        
    # Combine the found context into a single string
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    print("\n--- Phase 3: Generation ---")
    # We create a prompt template that enforces the LLM to only use our context
    prompt_template = PromptTemplate.from_template(
        "You are a helpful assistant. Use ONLY the following context to answer the user's question.\n"
        "If the answer isn't in the context, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=context, question=query)
    
    print("Sending prompt and context to Llama 3 for generation...")
    try:
        response = llm.invoke(prompt)
        print(f"\n🧠 AI Answer:\n{response.content}\n")
    except Exception as e:
        print(f"\n❌ Error connecting to Ollama: {e}")
        print("Did you remember to run 'docker exec -it ollama ollama pull llama3'?")

if __name__ == "__main__":
    main()
