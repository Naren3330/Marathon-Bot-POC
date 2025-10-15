import os
import time
import shutil
import asyncio
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
# FIX: Event Loop Management
# -------------------------------
def get_or_create_eventloop():
    """Get or create event loop for async operations"""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

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
            get_or_create_eventloop()
            
            parser = LlamaParse(
                api_key=LLAMAPARSE_API_KEY,
                result_type="markdown",
                verbose=True,
                language="en",
                num_workers=1,
                invalidate_cache=False,
                do_not_cache=False,
                gpt4o_mode=False,
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
                loop = get_or_create_eventloop()
                
                print(f"📄 Processing {filename} with LlamaParse...")
                parsed_docs = parser.load_data(file_path)
                
                for doc in parsed_docs:
                    # FIXED: Don't lowercase - preserve original text
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
                    # FIXED: Keep original casing
                    docs.append(Document(
                        page_content=content,
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
                
                # FIXED: Keep original casing
                docs.append(Document(
                    page_content=text,
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
# STEP 2: Split text into chunks (IMPROVED)
# -------------------------------
def create_chunks(docs):
    """Split documents into optimized chunks"""
    # FIXED: Smaller chunks for better semantic matching
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks = more precise retrieval
        chunk_overlap=100,  # More overlap for context continuity
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    
    # Add enhanced metadata
    for i, chunk in enumerate(chunks):
        # Add chunk position info
        chunk.metadata.update({
            "chunk_id": f"{chunk.metadata['source']}_chunk_{i}",
            "chunk_index": i,
            "chunk_length": len(chunk.page_content),
        })
    
    print(f"✅ Created {len(chunks)} chunks (avg size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars)")
    return chunks

#HYDE menthod for query enhancement
def enhance_query(query: str, llm=None) -> str:
    """
    Enhance user query using HyDE (Hypothetical Document Embeddings) method.
    Generates a hypothetical answer that matches the semantic space of actual documents.
    """
    # If no LLM provided, create one for query enhancement
    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    
    # HyDE prompt: Generate a hypothetical document that would answer this question
    hyde_prompt = f"""Given the following question about the Freshworks Chennai Marathon, 
write a hypothetical passage from a marathon information document that would answer this question.
Write as if you are the official marathon documentation.

Question: {query}

Hypothetical document passage:"""
    
    try:
        # Generate hypothetical answer
        hypothetical_doc = llm.predict(hyde_prompt)
        
        # Combine original query with hypothetical document for enhanced retrieval
        enhanced_query = f"{query}\n\nContext: {hypothetical_doc}"
        
        return enhanced_query
    
    except Exception as e:
        print(f"⚠️ HyDE enhancement failed: {e}")
        # Fallback to original query
        return query

# -------------------------------
# NEW: Query Enhancement
# -------------------------------
def enhance_query_manual(query: str) -> str:
    """Enhance user query for better retrieval"""
    # Common variations and expansions
    enhancements = {
        "start time": "start time race begin timing",
        "registration": "registration sign up register enroll",
        "5k": "5k 5km 5 km five kilometer",
        "10k": "10k 10km 10 km ten kilometer",
        "half marathon": "half marathon 21k 21km hm",
        "full marathon": "full marathon 42k 42km marathon fm",
        "route": "route path course track",
        "timing": "timing time start finish",
        "bib": "bib number race number",
        "race":"race date marathon date event day"
    }
    
    enhanced = query.lower()
    for key, expansion in enhancements.items():
        if key in enhanced:
            enhanced += " " + expansion
    
    return enhanced

# -------------------------------
# STEP 3: Create / Load Chroma DB (IMPROVED)
# -------------------------------
@st.cache_resource
def get_vectorstore():
    """Load existing vector store or create new one"""
    # FIXED: Use better embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000  # Batch size for API calls
    )
    
    # Check if vector store exists
    if os.path.exists(PERSIST_DIR):
        print(f"📦 Loading existing vector store from {PERSIST_DIR}...")
        try:
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings,
                # FIXED: Use L2 distance for OpenAI embeddings
                collection_metadata={"hnsw:space": "l2"}
            )
            print("✅ Vector store loaded successfully")
            return vectorstore
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
            print("🗑️ Attempting to rebuild vector store...")
            
            try:
                shutil.rmtree(PERSIST_DIR)
                print(f"✅ Deleted corrupted vector store at {PERSIST_DIR}")
            except Exception as delete_error:
                print(f"❌ Could not delete vector store: {delete_error}")
                return None
    
    # Create new vector store
    print(f"⚠️ Vector store not found at {PERSIST_DIR}")
    print("🔨 Creating new vector store...")
    
    docs = load_documents()
    if not docs:
        print("❌ No documents found in data directory!")
        return None
    
    chunks = create_chunks(docs)
    
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_metadata={"hnsw:space": "l2"}
        )
        print(f"✅ Vector store created and saved to {PERSIST_DIR}")
        return vectorstore
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return None

# -------------------------------
# STEP 4: Define the RAG chain (IMPROVED)
# -------------------------------
@st.cache_resource
def create_qa_chain(_vectorstore):
    """Create the RAG QA chain with enhanced retrieval"""
    
    # FIXED: Retrieve more chunks + use MMR for diversity
    retriever = _vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diverse results
        search_kwargs={
            "k": 8,  # Get more chunks
            "fetch_k": 20,  # Consider more candidates
            "lambda_mult": 0.5  # Balance relevance vs diversity
        }
    )
    
    # Use better model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    
    # FIXED: Much better prompt
    prompt_template_str = """You are an expert assistant for the Freshworks Chennai Marathon.

Your task is to answer questions accurately using ONLY the information provided in the context below.

IMPORTANT RULES:
1. Use ONLY information from the context - do not make up facts
2. If the context contains partial information, provide what you know and acknowledge what's missing
3. If the information is completely unavailable, say: "I don't have that specific information."
4. Be conversational and helpful - synthesize information from multiple parts of the context
5. When mentioning times, dates, or numbers, be precise
6. If asked about categories (5K, 10K, etc.), check all context chunks carefully

CONTEXT:
{context}

QUESTION: {question}

HELPFUL ANSWER:"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template_str,
    )
    print(prompt_template_str)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    print(qa_chain)
    return qa_chain

# -------------------------------
# UTILITY: Rebuild Vector Store
# -------------------------------
def rebuild_vector_store():
    """Delete existing vector store and trigger rebuild"""
    try:
        if os.path.exists(PERSIST_DIR):
            import gc
            gc.collect()
            time.sleep(0.5)
            shutil.rmtree(PERSIST_DIR)
            print(f"🗑️ Deleted {PERSIST_DIR}")
            time.sleep(0.5)
            return True
        else:
            print(f"⚠️ {PERSIST_DIR} does not exist")
            return False
    except PermissionError as e:
        print(f"❌ Permission error: {e}")
        print("💡 Try closing the app and manually deleting the folder")
        return False
    except Exception as e:
        print(f"❌ Error deleting vector store: {e}")
        return False

# -------------------------------
# STEP 5: Streamlit Chat UI (IMPROVED)
# -------------------------------
def main():
    st.set_page_config(
        page_title="Marathon Assistant",
        page_icon="🏃‍♂️",
        layout="wide"
    )
    
    st.title("🏃‍♂️ Freshworks Chennai Marathon Assistant")
    st.markdown("Ask me anything about the marathon — registration, race day, or partners!")
    
    if USE_LLAMAPARSE:
        st.success("✅ Enhanced parsing with LlamaParse enabled")
    else:
        st.info("📝 Using standard text loading")
    
    # Initialize vector store and QA chain from cache
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("❌ Failed to load knowledge base. Please check data directory.")
        st.stop()
    
    qa_chain = create_qa_chain(vectorstore)
    
    # Sidebar info
    with st.sidebar:
        st.header("📊 System Info")
        
        try:
            collection = st.session_state.vectorstore._collection
            doc_count = collection.count()
            st.metric("Documents in DB", doc_count)
        except:
            st.metric("Documents in DB", "N/A")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Ask natural questions: "When does the 5K start?"
        - Be specific: "What's included in registration?"
        - Try variations if first answer isn't perfect
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Maintenance")
        
        if st.button("🔄 Rebuild Vector Store", type="primary"):
            with st.spinner("Rebuilding vector store..."):
                # First, delete the files on disk
                success = rebuild_vector_store()
                
                if success:
                    # Then, clear the cached resources to force a reload
                    st.cache_resource.clear()
                    st.success("✅ Vector store deleted! Reloading app...")
                    time.sleep(1)
                    st.rerun()
                else:
                    # If deletion failed, no need to rerun
                    st.error("⚠️ Could not delete vector store. See console for details.")
    
    # Chat interface
    query = st.text_input("💬 Your question:", placeholder="e.g., What time does the 5K race start?")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        show_sources = st.checkbox("Show source documents", value=False)
        show_metrics = st.checkbox("Show performance metrics", value=True)
        use_query_enhancement = st.checkbox("Use query enhancement", value=True)
    
    if query:
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()

            with get_openai_callback() as cb:
                if use_query_enhancement:
                    st.info("🔍 Using HyDE for retrieval...")
                    
                    # 1. Generate hypothetical doc for retrieval
                    hyde_query = enhance_query(query)
                    
                    # 2. Use the QA chain's own retriever to fetch relevant documents
                    retriever = qa_chain.retriever
                    retrieved_docs = retriever.get_relevant_documents(hyde_query)
                    
                    # 3. Run the final step of the chain with the ORIGINAL query and retrieved docs
                    answer = qa_chain.combine_documents_chain.run(
                        input_documents=retrieved_docs,
                        question=query 
                    )
                    # Manually construct the result object to include source documents
                    result = {"result": answer, "source_documents": retrieved_docs}

                else:
                    # Standard retrieval using the original query
                    result = qa_chain.invoke({"query": query})
            
            end_time = time.time()
            query_time = round(end_time - start_time, 2)
        
        # Display answer
        st.write("### 💬 Answer")
        st.success(result["result"])
        
        # Show sources
        if show_sources and result["source_documents"]:
            st.write("### 📚 Retrieved Context")
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"📄 Chunk {i+1} from {doc.metadata['source']}"):
                    st.text(doc.page_content)
                    st.caption(f"Parser: {doc.metadata.get('parser', 'unknown')} | "
                             f"Length: {doc.metadata.get('chunk_length', 'N/A')} chars | "
                             f"Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
        
        # Show metrics
        if show_metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⏱️ Time", f"{query_time}s")
            col2.metric("🎯 Tokens", cb.total_tokens)
            col3.metric("💰 Cost", f"${cb.total_cost:.4f}")
            col4.metric("📄 Chunks", len(result["source_documents"]))
        
        # Console logging
        print("\n" + "="*60)
        print("📊 QUERY METRICS")
        print("="*60)
        print(f"Original Query: {query}")
        if use_query_enhancement:
            print(f"Enhanced Query: {hyde_query}")
        print(f"Answer: {result['result'][:200]}...")
        print(f"\nTotal Tokens: {cb.total_tokens}")
        print(f"Cost: ${cb.total_cost:.4f}")
        print(f"Query Time: {query_time} seconds")
        
        print("\n=== Retrieved Context Chunks ===")
        for i, doc in enumerate(result["source_documents"]):
            print(f"\nChunk {i+1} — Source: {doc.metadata['source']}")
            print("-" * 50)
            print(doc.page_content[:500])
            print("=" * 50)

if __name__ == "__main__":
    main()