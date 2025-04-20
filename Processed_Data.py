import os
from pathlib import Path

def process_data(filePath):
    try:
        abs_path = Path(os.path.abspath(filePath))
        with open(abs_path, "r") as file:
            return file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {abs_path}")
        print(f"Current working directory: {os.getcwd()}")
        return None

unprocessedTrain = process_data("EN/train")
unprocessedDevIn = process_data("EN/dev.in")
unprocessedDevOut = process_data("EN/dev.out")
ProcessedTrain = []
ProcessedDevIn = []
ProcessedDevOut = []

for line1 in unprocessedTrain:
    if len(line1)!=0:
        if line1 != '\n':
            temp = line1.rstrip().split(" ")
            value , tag = temp[0], temp[1]
        else:
            value, tag = "", "START"
        ProcessedTrain.append([value,tag])
        count+=1

for line2 in unprocessedDevIn:
    if len(line2)!=0:
        temp = line2.rstrip().split(" ")
        ProcessedDevIn.append(temp)
        count+=1

for line3 in unprocessedDevOut:
    if len(line3)!=0:
        temp = line3.rstrip().split(" ")
        value , tag = temp[0], temp[1]
        ProcessedTrain.append([value,tag])

def process_all_data():
    return {
        "ProcessedTrain": ProcessedTrain,
        "ProcessedDevIn": ProcessedDevIn,
        "ProcessedDevOut": ProcessedDevOut
    }

