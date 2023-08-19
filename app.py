import json
import os
import sys
import openai
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

def ask_question(question: str) -> str:
  load_dotenv()

  # set api key
  openai.api_key = os.environ.get("OPENAI_API_KEY")

  # load document
  context_file = "rides.txt"
  loader = TextLoader(context_file)

  # create index
  index = VectorstoreIndexCreator().from_loaders([loader])

  response = index.query(
    question,
    llm=ChatOpenAI()
  )
  print(f"<ai>: {response}")

  return response