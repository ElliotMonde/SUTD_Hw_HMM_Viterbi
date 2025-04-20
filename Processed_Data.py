import os
from pathlib import Path


def open_filepath(filePath):
    try:
        abs_path = Path(os.path.abspath(filePath))
        with open(abs_path, "r") as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {abs_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None


def process_all_data():

    unprocessedTrain = open_filepath("EN/train")
    unprocessedDevIn = open_filepath("EN/dev.in")
    unprocessedDevOut = open_filepath("EN/dev.out")
    ProcessedTrain = []
    ProcessedDevIn = []
    ProcessedDevOut = []

    START_S = "START"

    for line in unprocessedTrain:
        value, tag = "", START_S
        if len(line) != 0:
            temp = line.rstrip().split(" ")
            value, tag = temp[0], temp[1]
        ProcessedTrain.append((value, tag))

    for line in unprocessedDevIn:
        temp = "\n"
        if len(line) != 0:
            temp = line.rstrip().split(" ")
        ProcessedDevIn.append(temp)

    for line in unprocessedDevOut:
        value, tag = "", START_S
        if len(line) != 0:
            temp = line.rstrip().split(" ")
            value, tag = temp[0], temp[1]
        ProcessedTrain.append((value, tag))

    return {
        "ProcessedTrain": ProcessedTrain,
        "ProcessedDevIn": ProcessedDevIn,
        "ProcessedDevOut": ProcessedDevOut,
    }
