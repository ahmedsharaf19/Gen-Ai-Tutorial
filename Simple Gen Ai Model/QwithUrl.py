import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    """
)


# Define Model
llm = Ollama(model = 'gemma3:1b')
embedding = OllamaEmbeddings(model = 'mxbai-embed-large')
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=20, chunk_size=500)
output = StrOutputParser()
document_chain = create_stuff_documents_chain(llm, prompt)


## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

st.title('ChatBoot on Specific WEB')
url = st.text_input('URL need To Chat on : ')

if url:
    loader = WebBaseLoader(web_path=(url,))
    # Read Url Content
    docs = loader.load()
    
    # Chunks This Docs
    docs = text_splitter.split_documents(docs)
    
    # Create DB
    vectoredb = FAISS.from_documents(docs, embedding) 
    
    # Convert to retrieve
    retriever = vectoredb.as_retriever()
    
    question = st.text_input('Enter Your Qestion')


    if question:
        retiever_chain = create_retrieval_chain(retriever, document_chain)
        response = retiever_chain.invoke({'input': question})
        st.write(response['answer'])
