import os
import asyncio
import streamlit as st
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.set_page_config(page_title="Gemini RAG Chat", layout="wide")
st.title("Gemini RAG Chatbot")

if "retriever" not in st.session_state:
    st.session_state.retriever = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Ask something about your document:")

if uploaded_file and st.session_state.retriever is None:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        for i, page in enumerate(pages[:2]):
            st.write(f"Page {i+1} Preview:", page.page_content[:300])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        st.write(f"Total chunks created: {len(chunks)}")
        if chunks:
            st.write("🧩 Sample chunk:", chunks[0].page_content[:300])
        else:
            st.warning("⚠️ No chunks were created from the document.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(chunks, embedding=embeddings)

        st.session_state.retriever = vectorstore.as_retriever()
        st.success("✅ PDF processed and vectorstore created.")
    finally:
        os.remove(pdf_path)

if st.session_state.retriever:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3,)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an assistant for question-answering tasks. "
         "The responses should be detailed and informative. "
         "You will be provided with a context from a document. "
         "Use the retrieved context to answer the question. "
         "If unsure, say you don't know."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    if query:
        with st.spinner("Searching..."):
            response = rag_chain.invoke(query)
            answer = response.get("result") if isinstance(response, dict) else response
            st.markdown("### Answer")
            st.markdown(answer)

if st.button("Reset"):
    st.session_state.retriever = None
