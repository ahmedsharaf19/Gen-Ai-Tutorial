import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

prompt = ChatPromptTemplate.from_messages(
    [
     ('system', 'You are a helpful assistant. Please respond to the question asked'),
     ('user', 'Question: {question}')
    ]
)

st.title('LangChain Demo With Gemma Model')
input_text = st.text_input("What question you have in mind?")


# model
llm = Ollama(model = 'gemma3:1b')
parser = StrOutputParser()
chain = prompt|llm|parser

if input_text:
    st.write(chain.invoke({'question': input_text}))
