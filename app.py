import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Get Mistral API Key
# --------------------------------------------------

def get_mistral_api_key():
    """
    Get Mistral API key.
    Streamlit Cloud -> st.secrets
    Local machine -> .env
    """

    # First try Streamlit Cloud Secrets
    try:
        api_key = st.secrets.get("MISTRAL_API_KEY")

        if api_key:
            return api_key

    except Exception:
        pass

    # Then try .env
    api_key = os.getenv("MISTRAL_API_KEY")

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

    return ChatMistralAI(
        model="mistral-small-2603",
        api_key=st.secrets["MISTRAL_API_KEY"]
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
You are a helpful AI assistant.

Answer the question using ONLY the
provided context.

If the answer is not present in the
context, say:

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
                # Display AI Answer
                # ------------------------------------------

                st.subheader(
                    "🤖 AI Answer"
                )

                st.write(
                    response.content
                )


                # ------------------------------------------
                # Show retrieved chunks
                # ------------------------------------------

                with st.expander(
                    "🔍 View Retrieved Chunks"
                ):

                    for i, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        st.markdown(
                            f"**Chunk {i}**"
                        )

                        st.write(
                            doc.page_content
                        )


            except Exception as e:

                st.error(
                    "❌ Mistral API Error"
                )

                st.code(
                    str(e)
                )