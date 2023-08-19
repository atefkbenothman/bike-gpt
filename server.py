from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app import ask_question

app = FastAPI()

origins = [
  "http://localhost",
  "http://localhost:3000"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_methods=["*"],
  allow_headers=["*"]
)

@app.post("/chat")
async def chat(data: dict):
  print(data)
  question = data.get("question")
  question += ". respond using html"
  response: str = ask_question(question)
  data = {
    "data": response
  }
  return data