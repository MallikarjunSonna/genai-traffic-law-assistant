import streamlit as st
from dotenv import load_dotenv
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Loading environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit set up
st.set_page_config(page_title="Traffic Law Assistant", page_icon="🛵")
st.title("🛵 Indian Traffic Law GenAI Assistant")

# loading the PDF of the Motor Vehicles Act used in the project
PDF_FILE = "2202011053641.pdf"
loader = PyPDFLoader(PDF_FILE)
pages = loader.load()

# Spliting into chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(pages)

# converting chunks into vectors by using OpenAIEmbeddings and store in FAISS VectorDatabase
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context:
<context>
{context}
</context>

Question: {input}
""")

# LLM and Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

#  Streamlit Input
query = st.text_input("Ask your legal question about Indian traffic laws:", "")

if query:
    with st.spinner("Thinking..."):
        response = retriever_chain.invoke({"input": query})
        st.markdown("### 📌 Answer")
        st.write(response["answer"])
