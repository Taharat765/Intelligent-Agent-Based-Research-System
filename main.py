from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

@app.get("/")
async def home():
    return {"status": "running"}

@app.get("/test")
async def test():
    response = llm.invoke("Say hello")
    return {"response": response.content}