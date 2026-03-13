import streamlit as st
import time
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate

# URLs for services inside the Docker Network
OLLAMA_URL = "http://ollama:11434"
QDRANT_URL = "http://qdrant:6333"
COLLECTION_NAME = "learning_rag_ui"

st.set_page_config(page_title="Local RAG Tutorial", page_icon="🤖", layout="wide")

st.title("🤖 Local RAG Experience")
st.markdown("A 100% local, privacy-respecting RAG utilizing Ollama, Langchain, and Qdrant.")

@st.cache_resource
def get_vector_store():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL,
    )
    # Connect to the vector store (without force_recreate so we can query existing state)
    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
    )
    return store, embeddings

# Get cached vector store
try:
    vector_store, embeddings = get_vector_store()
except Exception as e:
    # If the collection doesn't exist yet, we catch the exception and init it empty
     embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL,
     )
     vector_store = QdrantVectorStore.from_documents(
        [],
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


# Sidebar for Ingestion Phase
with st.sidebar:
    st.header("🗂️ Phase 1: Ingestion")
    st.markdown("Add data into the Vector Database.")
    
    tab1, tab2 = st.tabs(["Upload Document", "Paste Text"])
    
    with tab1:
        st.subheader("Upload PDF or TXT")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"], label_visibility="collapsed")
        if st.button("Learn Document", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner(f"Reading {uploaded_file.name}..."):
                    text_content = ""
                    # If it's a PDF, extract via pypdf
                    if uploaded_file.name.lower().endswith('.pdf'):
                        import pypdf
                        pdf_reader = pypdf.PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                text_content += text + "\n"
                    # If it's a text file
                    else:
                        text_content = uploaded_file.getvalue().decode("utf-8")
                    
                    if text_content.strip():
                        # We chunk the document so the Vector DB can search effectively
                        from langchain_text_splitters import RecursiveCharacterTextSplitter
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_text(text_content)
                        docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
                        
                        vector_store.add_documents(docs)
                        st.success(f"Learned {len(docs)} chunks from {uploaded_file.name}!")
                    else:
                        st.warning("Could not extract text from document.")
            else:
                st.warning("Please browse for a file first.")

    with tab2:
        st.subheader("Paste Raw Text")
        new_knowledge = st.text_area("Enter some text to learn:", height=150, placeholder="The secret password is 'PINEAPPLE'...")
        if st.button("Learn Text", use_container_width=True):
            if new_knowledge:
                with st.spinner("Embedding and saving to Qdrant..."):
                    docs = [Document(page_content=new_knowledge)]
                    vector_store.add_documents(docs)
                    st.success("Successfully ingested into Vector DB!")
            else:
                st.warning("Please enter some text first.")
            
    st.divider()
    if st.button("🗑️ Clear Vector Database"):
         with st.spinner("Deleting all data..."):
             QdrantVectorStore.from_documents(
                [],
                embeddings,
                url=QDRANT_URL,
                collection_name=COLLECTION_NAME,
                force_recreate=True,
             )
             st.success("Database cleared!")

# Main window for Retrieval & Generation
st.header("💬 Phase 2 & 3: Retrieval and Generation")

query = st.text_input("Ask a question about your documents:")
if st.button("Submit Query"):
    if query:
        # Phase 2: Retrieval
        with st.status("🔍 Searching Vector Database...", expanded=True) as status:
            retrieved_docs = vector_store.similarity_search(query, k=3)
            
            st.write("Retrieved the following chunks:")
            for i, doc in enumerate(retrieved_docs):
                 st.info(f"**Chunk {i+1}:** {doc.page_content}")
            
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            status.update(label="✅ Retrieval Complete", state="complete", expanded=False)
        
        # Phase 3: Generation
        with st.spinner("🧠 Generating answer with Llama 3..."):
            llm = ChatOllama(
                model="llama3",
                base_url=OLLAMA_URL,
                temperature=0.2,
            )
            
            prompt_template = PromptTemplate.from_template(
                "You are a helpful assistant. Use ONLY the following context to answer the user's question.\n"
                "If the answer isn't in the context, say 'I don't know'.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            prompt = prompt_template.format(context=context, question=query)
            
            try:
                response = llm.invoke(prompt)
                st.markdown("### 🧠 AI Response")
                st.write(response.content)
            except Exception as e:
                st.error(f"Error communicating with Ollama: {e}")
                st.info("Did you remember to pull the models via `docker exec -it ollama ollama pull llama3`?")
    else:
        st.warning("Please enter a question.")
