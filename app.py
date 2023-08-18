#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
import argparse
import json
import os
import sys
import openai


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
    moving_time: {activity["moving_time"]}
    elapsed_time: {activity["elapsed_time"]}
    total_elevation_gain: {activity["total_elevation_gain"]}
    start_date: {activity["start_date_local"]}
    achievement_count: {activity["achievement_count"]}
    average_speed: {activity["average_speed"]}
    max_speed: {activity["max_speed"]}
    average_temp: {activity["average_temp"]}
    average_watts: {activity["average_watts"]}
    kilojoules: {activity["kilojoules"]}
    elev_high: {activity["elev_high"]}
    elev_low: {activity["elev_low"]}
    personal_records: {activity["pr_count"]}
    """
    output.append(data)

  with open(output_file, "a") as f:
    for activity in output:
      f.write(activity)

  print("done")
  return


def convert_meters_to_miles(meters: float) -> str:
  return f"{meters * 0.000621371:.2f}"


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

  response = index.query(
    args.prompt,
    llm=ChatOpenAI(),
  )
  print(f"<ai>: {response}")