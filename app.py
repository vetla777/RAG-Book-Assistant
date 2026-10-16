import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Get Mistral API Key
# --------------------------------------------------

def get_gemini_api_key():
    """
    Get Mistral API key.
    Streamlit Cloud -> st.secrets
    Local machine -> .env
    """

    # First try Streamlit Cloud Secrets
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")

        if api_key:
            return api_key

    except Exception:
        pass

    # Then try .env
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    return None


# --------------------------------------------------
# Create a writable temporary directory for ChromaDB
# --------------------------------------------------

if "chroma_path" not in st.session_state:
    st.session_state["chroma_path"] = tempfile.mkdtemp(
        prefix="rag_book_assistant_"
    )

CHROMA_PATH = st.session_state["chroma_path"]

COLLECTION_NAME = "rag_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Book Assistant")

st.write(
    "Upload a PDF and ask questions from the document"
)


# --------------------------------------------------
# Load HuggingFace Embedding Model
# --------------------------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# --------------------------------------------------
# Load Mistral LLM
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY")
    )


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF book",
    type=["pdf"]
)


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

if uploaded_file:

    st.success("PDF uploaded successfully!")

    if st.button("Create Vector Database"):

        try:

            with st.status(
                "Processing document...",
                expanded=True
            ) as status:

                # ------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.getvalue()
                    )

                    file_path = tmp_file.name


                # ------------------------------------------
                # Load PDF
                # ------------------------------------------

                st.write("📄 Loading PDF...")

                loader = PyPDFLoader(
                    file_path
                )

                docs = loader.load()

                st.write(
                    f"📄 Pages loaded: {len(docs)}"
                )


                # ------------------------------------------
                # Split document
                # ------------------------------------------

                st.write(
                    "✂️ Splitting document..."
                )

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    docs
                )

                st.write(
                    f"📦 Number of chunks: {len(chunks)}"
                )


                # ------------------------------------------
                # Create a fresh Chroma directory
                # ------------------------------------------

                if os.path.exists(CHROMA_PATH):

                    st.write(
                        "🗑️ Removing old vector database..."
                    )

                    shutil.rmtree(
                        CHROMA_PATH,
                        ignore_errors=True
                    )


                # Recreate writable directory

                os.makedirs(
                    CHROMA_PATH,
                    exist_ok=True
                )


                # ------------------------------------------
                # Load embedding model
                # ------------------------------------------

                st.write(
                    "🤖 Loading HuggingFace embedding model..."
                )

                embeddings = get_embeddings()


                # ------------------------------------------
                # Create ChromaDB
                # ------------------------------------------

                st.write(
                    "🧠 Creating embeddings and storing in ChromaDB..."
                )

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=COLLECTION_NAME,
                    persist_directory=CHROMA_PATH
                )


                # ------------------------------------------
                # Store vector database in session
                # ------------------------------------------

                st.session_state["vectorstore"] = vectorstore

                st.session_state["db_ready"] = True


                # ------------------------------------------
                # Update status
                # ------------------------------------------

                status.update(
                    label="Vector database created successfully!",
                    state="complete"
                )


            st.success(
                f"✅ Vector database created with {len(chunks)} chunks!"
            )


        except Exception as e:

            st.error(
                f"❌ Error while creating vector database:\n\n{e}"
            )


        finally:

            # ------------------------------------------
            # Remove temporary PDF
            # ------------------------------------------

            if (
                "file_path" in locals()
                and os.path.exists(file_path)
            ):

                os.remove(
                    file_path
                )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.get(
    "db_ready",
    False
):

    st.divider()

    st.subheader(
        "💬 Ask Questions From the Book"
    )


    query = st.text_input(
        "Enter your question"
    )


    if query:

        with st.spinner(
            "Searching the document..."
        ):

            try:

                # ------------------------------------------
                # Get vector database
                # ------------------------------------------

                vectorstore = st.session_state[
                    "vectorstore"
                ]


                # ------------------------------------------
                # MMR Retriever
                # ------------------------------------------

                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )


                # ------------------------------------------
                # Retrieve relevant documents
                # ------------------------------------------

                retrieved_docs = retriever.invoke(
                    query
                )


                # ------------------------------------------
                # Create context
                # ------------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # ------------------------------------------
                # Prompt
                # ------------------------------------------

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
You are a clear, helpful AI assistant that explains concepts from the provided book.

Answer the question using ONLY the provided context. Give a complete, structured explanation — not just one sentence.

Structure your answer clearly:
- Start with a direct definition or explanation
- Include examples with numbers where relevant (e.g., "If you have 1000 samples and batch size 100, you get 10 iterations per epoch")
- Mention key relationships between terms (epoch vs batch vs iteration)
- If the source chunk contains an example, include it
- Cite sources with [Chunk 1], [Chunk 2], etc.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Do not make up information.
"""
                        ),
                        (
                            "human",
                            """
Context:

{context}

Question:

{question}
"""
                        )
                    ]
                )


                # ------------------------------------------
                # Create final prompt
                # ------------------------------------------

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )


                # ------------------------------------------
                # Mistral LLM
                # ------------------------------------------

                llm = get_llm()


                # ------------------------------------------
                # Generate answer
                # ------------------------------------------

                response = llm.invoke(
                    final_prompt
                )


                # ------------------------------------------
                # Extract actual answer — handle AIMessage, list, and dict responses
                # ------------------------------------------
                raw_response = response
                if isinstance(raw_response, list):
                    parts = []
                    for item in raw_response:
                        if hasattr(item, "content"):
                            inner = item.content
                            if isinstance(inner, list):
                                parts.extend(str(x) for x in inner)
                            else:
                                parts.append(str(inner))
                        else:
                            parts.append(str(item))
                    answer_text = "\n".join(parts)
                elif hasattr(raw_response, "content"):
                    inner = raw_response.content
                    if isinstance(inner, list):
                        answer_text = "\n".join(str(x) for x in inner)
                    else:
                        answer_text = str(inner)
                else:
                    answer_text = str(raw_response)

                # Clean up AIMessage wrapper / dict responses
                if isinstance(answer_text, str):
                    import ast, re
                    dict_text_match = re.search(
                        r"'text'\s*:\s*'(.+?)'\s*,\s*'extras'",
                        answer_text, re.DOTALL
                    )
                    if dict_text_match:
                        answer_text = dict_text_match.group(1)
                        answer_text = answer_text.replace("\\n", "\n").replace("\\t", "\t")
                    else:
                        dict_text_match2 = re.search(
                            r'"text"\s*:\s*"(.+?)"\s*,\s*"extras"',
                            answer_text, re.DOTALL
                        )
                        if dict_text_match2:
                            answer_text = dict_text_match2.group(1)
                    try:
                        parsed = ast.literal_eval(answer_text)
                        if isinstance(parsed, dict):
                            if "text" in parsed and isinstance(parsed["text"], str):
                                answer_text = parsed["text"]
                            elif "content" in parsed and isinstance(parsed["content"], str):
                                answer_text = parsed["content"]
                    except Exception:
                        pass
                    m = re.search(r"content='(.*?)'", answer_text, re.DOTALL)
                    if m:
                        answer_text = m.group(1)
                    else:
                        m = re.search(r'content="(.*?)"', answer_text, re.DOTALL)
                        if m:
                            answer_text = m.group(1)
                    answer_text = re.sub(r"AIMessage\([^)]*\)", "", answer_text)
                    answer_text = answer_text.strip()
                else:
                    answer_text = str(answer_text)

                # ------------------------------------------
                # Display AI Answer — structured with citation legend
                # ------------------------------------------
                citation_legend = " ".join(
                    f"<span style='display:inline-block;background:#3d2b8e;color:#e8dff5;padding:0.2rem 0.55rem;border-radius:20px;font-weight:700;font-size:0.8rem;border:1px solid #8b6cf0;'>[{i+1}] Page {doc.metadata.get('page', '?')}</span>"
                    for i, doc in enumerate(retrieved_docs)
                )

                st.subheader("🤖 AI Answer")

                st.markdown(
                    f"""
                    <div style="background:#1e1e2e;padding:1.5rem;border-radius:12px;border-left:5px solid #8b6cf0;box-shadow:0 2px 10px rgba(139,108,240,0.2);color:#f0f0f5;line-height:1.65;font-size:1.05rem;">
                        <div style="margin-bottom:0.75rem;display:flex;gap:0.4rem;flex-wrap:wrap;">
                            {citation_legend}
                        </div>
                        <div style="font-family:system-ui,-apple-system,sans-serif;">{answer_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                # ------------------------------------------
                # Show retrieved chunks — improved card layout
                # ------------------------------------------

                with st.expander("📚 View Retrieved Sources (with page numbers)", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        for i, doc in enumerate(retrieved_docs, start=1):
                            page = doc.metadata.get("page", "N/A")
                            st.markdown(
                                f"""
                                <div style="background:#26263a;border:1px solid #383850;border-left:5px solid #8b6cf0;padding:1rem;border-radius:8px;margin:0.6rem 0;box-shadow:0 1px 3px rgba(139,108,240,0.15);color:#e8e8f0;line-height:1.5;font-size:0.95rem;">
                                    <strong style="color:#6C63FF;font-size:1rem;">📄 Chunk {i}</strong> &nbsp;|&nbsp; <span style="color:#555;font-size:0.85rem;">Page {page}</span>
                                    <hr style="margin:0.6rem 0;border:none;border-top:1px solid #dde0e6;">
                                    <div>{doc.page_content}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    with col2:
                        st.info(
                            "💡 Each chunk is a 1000-character section from the PDF. "
                            "Use the citation badges in the answer to see which chunk supported each point."
                        )


            except Exception as e:

                st.error(
                    "❌ Gemini API Error"
                )

                st.code(
                    str(e)
                )