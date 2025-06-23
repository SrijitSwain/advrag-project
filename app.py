import os
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS

# Apply nested asyncio event loop (for Streamlit)
nest_asyncio.apply()

# Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq LLM setup
def get_groq_llm(model_name: str):
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model_name
    )

# Create FAISS vector store with HuggingFace Embeddings (free, no API key needed)
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Small, fast model
    return FAISS.from_documents(docs, embeddings)

# Process PDF file into document chunks
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# Process URL into document chunks
def process_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# DuckDuckGo search function
def duckduckgo_search(query, max_results=3):
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
    return results

# Streamlit App
st.title("🧠 Srijit’s AI-Powered Knowledge Search & Chat Tool 🚀")

st.markdown("""
## 📚 Intelligent Knowledge Base & Internet Assistant  
Built with 💻 Langchain, Groq LLMs, FAISS & DuckDuckGo Search.

---

#### 👨‍💻 Developed by **Srijit Swain**
""")

# Valid Groq Models only (no Mixtral)
model_option = st.selectbox("Select Groq Model:", ["llama3-70b-8192", "llama3-8b-8192", "gemma-7b-it"])
llm = get_groq_llm(model_option)

# Upload PDF
st.header("📄 Upload PDF to Knowledge Base")
pdf_file = st.file_uploader("Choose PDF", type="pdf")
pdf_docs = []

if pdf_file:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    pdf_docs = process_pdf("temp.pdf")
    st.success("PDF processed and added to knowledge base.")

# Input URL
st.header("🔗 Add URL to Knowledge Base")
url_input = st.text_input("Enter URL:")
url_docs = []

if st.button("Process URL"):
    if url_input:
        url_docs = process_url(url_input)
        st.success("URL processed and added to knowledge base.")
    else:
        st.error("Please provide a valid URL.")

# Build FAISS Vectorstore
all_docs = pdf_docs + url_docs
vectorstore = None
if all_docs:
    vectorstore = create_vectorstore(all_docs)

# Question Answering
st.header("❓ Ask a Question from Knowledge Base")

query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = "No documents in the knowledge base."

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a helpful AI assistant. Answer based on the provided context.

        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"context": context, "question": query})
    st.write("💡 Answer:", response)

# DuckDuckGo Search
st.header("🌐 Internet Search (DuckDuckGo)")
search_query = st.text_input("Enter search query:")
if st.button("Search Internet"):
    if search_query:
        results = duckduckgo_search(search_query)
        st.write("🔍 DuckDuckGo Results:")
        for res in results:
            st.write(f"**Title:** {res.get('title')}\n**URL:** {res.get('href')}\n**Body:** {res.get('body')}\n---")
    else:
        st.error("Please enter a search query.")
