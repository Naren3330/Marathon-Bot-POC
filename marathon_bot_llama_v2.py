import os
import time
import streamlit as st
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# CONFIGURATION
# -------------------------------
DATA_DIR = "data"
PERSIST_DIR = "marathon_kb_enhanced"
LLAMAPARSE_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
USE_LLAMAPARSE = LLAMAPARSE_API_KEY is not None and LLAMAPARSE_API_KEY.strip() != ""

# -------------------------------
# STEP 1: Load Knowledge Base with LlamaParse
# -------------------------------
def load_documents():
    """Load documents using LlamaParse or fallback methods"""
    docs = []
    
    # Initialize LlamaParse if available
    parser = None
    if USE_LLAMAPARSE:
        try:
            parser = LlamaParse(
                api_key=LLAMAPARSE_API_KEY,
                result_type="markdown",
                verbose=True,
                language="en",
                num_workers=1,  # Single worker for stability
                invalidate_cache=False,
                do_not_cache=False,
                gpt4o_mode=False,  # Faster processing
            )
            print("✅ LlamaParse initialized successfully")
        except Exception as e:
            print(f"⚠️ LlamaParse initialization failed: {e}")
            print("📝 Using fallback loading methods")
            parser = None
    else:
        print("📝 LLAMA_CLOUD_API_KEY not set. Using fallback loading.")
    
    # Load all files from data directory
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        
        # Try LlamaParse first if available
        if parser and filename.endswith((".pdf", ".docx", ".doc", ".pptx", ".txt")):
            try:
                print(f"📄 Processing {filename} with LlamaParse...")
                parsed_docs = parser.load_data(file_path)
                
                for doc in parsed_docs:
                    docs.append(Document(
                        page_content=doc.text,
                        metadata={
                            "source": filename,
                            "parser": "llamaparse",
                            "file_type": filename.split(".")[-1]
                        }
                    ))
                print(f"✅ Parsed {filename}")
                continue
                
            except Exception as e:
                print(f"⚠️ LlamaParse failed for {filename}: {e}")
                print(f"📝 Trying fallback method...")
        
        # Fallback: Standard text loading
        if filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    docs.append(Document(
                        page_content=content.lower(),  # Normalize
                        metadata={
                            "source": filename,
                            "parser": "text",
                            "file_type": "txt"
                        }
                    ))
                print(f"✅ Loaded {filename} with text loader")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")
        
        # Fallback: PDF loading with PyPDF2
        elif filename.endswith(".pdf"):
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                docs.append(Document(
                    page_content=text.lower(),
                    metadata={
                        "source": filename,
                        "parser": "pypdf2",
                        "file_type": "pdf"
                    }
                ))
                print(f"✅ Loaded {filename} with PyPDF2")
            except Exception as e:
                print(f"❌ Failed to load PDF {filename}: {e}")
    
    print(f"\n✅ Total documents loaded: {len(docs)}")
    return docs

# -------------------------------
# STEP 2: Split text into chunks
# -------------------------------
def create_chunks(docs):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": f"{chunk.metadata['source']}_chunk_{i}",
            "chunk_index": i
        })
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# -------------------------------
# STEP 3: Create / Load Chroma DB
# -------------------------------
def get_vectorstore():
    """Load existing vector store or create new one"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if vector store exists
    if os.path.exists(PERSIST_DIR):
        print(f"📦 Loading existing vector store from {PERSIST_DIR}...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        print("✅ Vector store loaded successfully")
        return vectorstore
    else:
        print(f"⚠️ Vector store not found at {PERSIST_DIR}")
        print("🔨 Creating new vector store...")
        
        # Load documents
        docs = load_documents()
        if not docs:
            print("❌ No documents found in data directory!")
            return None
        
        # Create chunks
        chunks = create_chunks(docs)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        print(f"✅ Vector store created and saved to {PERSIST_DIR}")
        return vectorstore

# -------------------------------
# STEP 4: Define the RAG chain
# -------------------------------
def create_qa_chain(vectorstore):
    """Create the RAG QA chain"""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Define prompt
    prompt_template_str = """You are a helpful chatbot trained exclusively on Freshworks Chennai Marathon data.
Use ONLY the provided context to answer the user's question.
If the information is not available in the context, reply exactly with:
"I'm sorry, I don't have that information."

Context:
{context}

Question:
{question}

Answer:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_str,
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# -------------------------------
# STEP 5: Streamlit Chat UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="Marathon Assistant",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Freshworks Chennai Marathon Assistant")
    st.markdown("Ask me anything about the marathon — registration, race day, or partners!")
    
    # Show parsing method being used
    if USE_LLAMAPARSE:
        st.success("✅ Enhanced parsing with LlamaParse enabled")
    else:
        st.info("📝 Using standard text loading")
    
    # Initialize vector store
    if 'vectorstore' not in st.session_state:
        with st.spinner("🔄 Loading knowledge base..."):
            st.session_state.vectorstore = get_vectorstore()
            
            if st.session_state.vectorstore is None:
                st.error("❌ Failed to load knowledge base. Please check data directory.")
                st.stop()
            
            st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 System Info")
        
        # Check vector store stats
        try:
            collection = st.session_state.vectorstore._collection
            doc_count = collection.count()
            st.metric("Documents in DB", doc_count)
        except:
            st.metric("Documents in DB", "N/A")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask specific questions
        - Mention race categories (5K, 10K, etc.)
        - Ask about registration, timing, routes
        """)
        
        if st.button("🔄 Rebuild Vector Store"):
            st.info("To rebuild, delete the 'marathon_kb_enhanced' folder and restart the app.")
    
    # Chat interface
    query = st.text_input("💬 Your question:", placeholder="e.g., What time does the race start?")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        show_sources = st.checkbox("Show source documents", value=False)
        show_metrics = st.checkbox("Show performance metrics", value=False)
    
    if query:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            with get_openai_callback() as cb:
                result = st.session_state.qa_chain.invoke({"query": query})
            
            end_time = time.time()
            query_time = round(end_time - start_time, 2)
        
        # Display answer
        st.write("### 💬 Answer")
        st.success(result["result"])
        
        # Show sources
        if show_sources and result["source_documents"]:
            st.write("### 📚 Sources")
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"Source {i+1}: {doc.metadata['source']}"):
                    st.text(doc.page_content[:500] + "...")
                    st.caption(f"Parser: {doc.metadata.get('parser', 'unknown')} | "
                             f"Chunk: {doc.metadata.get('chunk_index', 'N/A')}")
        
        # Show metrics
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ Time", f"{query_time}s")
            col2.metric("🎯 Tokens", cb.total_tokens)
            col3.metric("💰 Cost", f"${cb.total_cost:.4f}")
            col4.metric("📄 Chunks", len(result["source_documents"]))
        
        # Console logging (for debugging)
        print("\n" + "="*60)
        print("📊 QUERY METRICS")
        print("="*60)
        print(f"Query: {query}")
        print(f"Answer: {result['result'][:200]}...")
        print(f"\nTotal Tokens: {cb.total_tokens}")
        print(f"Input Tokens: {cb.prompt_tokens}")
        print(f"Output Tokens: {cb.completion_tokens}")
        print(f"Cost: ${cb.total_cost:.4f}")
        print(f"Query Time: {query_time} seconds")
        
        print("\n=== Retrieved Context Chunks ===")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nChunk {i+1} — Source: {doc.metadata['source']}")
            print("-" * 50)
            print(doc.page_content[:1000])
            print("=" * 50)

if __name__ == "__main__":
    main()