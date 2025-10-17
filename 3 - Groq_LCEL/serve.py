import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn

# prompt
systemmsg = "Translate From English to {lang}"
prompt = ChatPromptTemplate.from_messages([
    ('system', systemmsg),
    ('user', '{mssg}')
])

parser = StrOutputParser()

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = ChatGroq(model = 'Gemma2-9b-It', groq_api_key=groq_api_key)

chain = prompt|model|parser

app = FastAPI(
    title='Gemma API',
    version='1.0',
    description='A simple API server using Langchain runnable interfaces'
)


add_routes(
    app,
    runnable = chain,
    path = '/chain',
)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
