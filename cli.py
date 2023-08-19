#!/usr/bin/env python3
import argparse
import json
import os
import sys
import openai
from dotenv import load_dotenv
from datetime import datetime
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def convert_json_to_humanreadable():
  """
  convert activities json data to a human readable format
  that can be parsed by langchain.
  """
  print("converting json data to human readable format")

  input_file = "rides.json"
  output_file = "rides.txt"

  if os.path.exists(output_file):
    os.remove(output_file)

  with open(input_file, "r") as f:
    activities = json.load(f)

  output = []
  for activity in activities:
    data = f"""
    name: {activity["name"]}
    type: {activity["type"]}
    distance: {convert_meters_to_miles(activity["distance"])} miles
    moving_time: {convert_seconds_to_hours(activity["moving_time"])}
    elapsed_time: {convert_seconds_to_hours(activity["elapsed_time"])}
    total_elevation_gain: {convert_meters_to_feet(activity["total_elevation_gain"])} feet
    start_date: {format_datetime(activity["start_date_local"])}
    achievement_count: {activity["achievement_count"]} achievements
    average_speed: {convert_mps_to_mph(activity["average_speed"])} mph
    max_speed: {convert_mps_to_mph(activity["max_speed"])} mph
    average_temp: {convert_celsius_to_fahrenheit(activity["average_temp"])} f
    average_watts: {activity["average_watts"]} watts
    kilojoules: {activity["kilojoules"]} kj
    elev_high: {convert_meters_to_feet(activity["elev_high"])} feet
    elev_low: {convert_meters_to_feet(activity["elev_low"])} feet
    personal_records: {activity["pr_count"]} records
    """
    output.append(data)

  with open(output_file, "a") as f:
    for activity in output:
      f.write(activity)

  print("done")
  return


def convert_meters_to_miles(meters: float) -> str:
  return f"{meters * 0.000621371:.2f}"


def convert_seconds_to_hours(seconds: float) -> str:
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  return f"{hours} hours {minutes} mins"


def convert_meters_to_feet(meters: float) -> str:
  return str(int(meters * 3.28084))


def format_datetime(date: str) -> str:
  input_format = "%Y-%m-%dT%H:%M:%SZ"
  output_format = "%b %dth, %Y %I:%M%p"
  parsed_date = datetime.strptime(date, input_format)
  formatted_date = parsed_date.strftime(output_format)
  return formatted_date


def convert_mps_to_mph(mps: float) -> str:
  return f"{mps * 2.23694:.2f}"


def convert_celsius_to_fahrenheit(celsius: int) -> str:
  return str(int((celsius * 9/5) + 32))


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



if __name__ == "__main__":
  load_dotenv()

  # set api key
  openai.api_key = os.environ.get("OPENAI_API_KEY")

  parser = argparse.ArgumentParser(
    description="ask questions about your cycling activities",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "--convert",
    action="store_true",
    help="convert json data to human-readable format"
  )
  parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="question you want to ask"
  )
  args = parser.parse_args()

  if args.convert:
    convert_json_to_humanreadable()

  if args.prompt is None:
    print("remember to ask a question using the --prompt flag")
    sys.exit(1)

  context_file = "rides.txt"
  # check if we have data we can use as context
  if not os.path.exists(context_file):
    print("convert your ride data using the --convert flag first")
    sys.exit(1)

  # load document
  loader = TextLoader(context_file)

  # create index
  index = VectorstoreIndexCreator().from_loaders([loader])

  # setup chain
  chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1})
  )

  # setup chat
  chat_history = []

  prompt = args.prompt
  while True:
    if not prompt:
      prompt = input("<user>: ")
    if prompt in ["quit", "q", "exit"]:
      sys.exit(1)

    response = chain({"question": prompt, "chat_history": chat_history})
    print(f"<ai>: {response['answer']}")

    chat_history.append((prompt, response["answer"]))
    prompt = None

  # response = index.query(
  #   args.prompt,
  #   llm=ChatOpenAI()
  # )
  # print(f"<ai>: {response}")