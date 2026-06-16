import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("google_api_key")

st.title("Equity Research Dashboard")

st.sidebar.title("URL'S for Equity Research")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key,
    max_tokens=500
)

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000,
        )
    main_placeholder.text("Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
        )
    vectorstore_openai = FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    vectorstore_openai.save_local("faiss_index")
    main_placeholder.text("Vector database saved successfully!")


query = main_placeholder.text_input("Enter your question here:")

if query:

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    if os.path.exists("faiss_index"):

        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain.invoke(
            {"question": query},
        )

        st.header("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            st.write(result["sources"])



