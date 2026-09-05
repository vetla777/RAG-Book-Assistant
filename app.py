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

CHROMA_PATH = "./chroma_db"
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
st.write("Upload a PDF and ask questions from the document")


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
# Load Mistral
# --------------------------------------------------

@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model="mistral-small-2603"
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

                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name

                st.write("📄 Loading PDF...")

                # ------------------------------------------
                # Load PDF
                # ------------------------------------------

                loader = PyPDFLoader(file_path)

                docs = loader.load()

                st.write(
                    f"📄 Pages loaded: {len(docs)}"
                )

                # ------------------------------------------
                # Split document
                # ------------------------------------------

                st.write("✂️ Splitting document...")

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                st.write(
                    f"📦 Number of chunks: {len(chunks)}"
                )

                # ------------------------------------------
                # Delete old Chroma database
                # ------------------------------------------

                if os.path.exists(CHROMA_PATH):

                    st.write(
                        "🗑️ Removing old vector database..."
                    )

                    shutil.rmtree(CHROMA_PATH)

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
                # Store in session
                # ------------------------------------------

                st.session_state["vectorstore"] = vectorstore
                st.session_state["db_ready"] = True

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

            # Remove temporary PDF
            if "file_path" in locals() and os.path.exists(file_path):
                os.remove(file_path)


# --------------------------------------------------
# Load Existing Vector Database
# --------------------------------------------------

if (
    not st.session_state.get("db_ready", False)
    and os.path.exists(CHROMA_PATH)
):

    try:

        embeddings = get_embeddings()

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        st.session_state["vectorstore"] = vectorstore
        st.session_state["db_ready"] = True

    except Exception as e:

        st.warning(
            f"Existing Chroma database could not be loaded: {e}"
        )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.get("db_ready", False):

    st.divider()

    st.subheader("💬 Ask Questions From the Book")

    query = st.text_input(
        "Enter your question"
    )

    if query:

        with st.spinner("Searching the document..."):

            vectorstore = st.session_state["vectorstore"]

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

            retrieved_docs = retriever.invoke(query)

            # ------------------------------------------
            # Create Context
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

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            # ------------------------------------------
            # Mistral
            # ------------------------------------------

            llm = get_llm()

            response = llm.invoke(
                final_prompt
            )

        st.subheader("🤖 AI Answer")

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