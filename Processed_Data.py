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

unprocessedTrain = process_data("EN/EN/train")
unprocessedDevIn = process_data("EN/EN/dev.in")
unprocessedDevOut = process_data("EN/EN/dev.out")
ProcessedTrain = []
ProcessedDevIn = []
ProcessedDevOut = []
count = 0
for line1 in unprocessedTrain:
    if len(line1)!=0 and line1!="\n":
        temp = line1.rstrip().split(" ")
        value , tag = temp[0], temp[1]
        ProcessedTrain.append([value,tag])
        count+=1
count = 0

for line2 in unprocessedDevIn:
    if len(line2)!=0 and line2!="\n":
        temp = line2.rstrip().split(" ")
        ProcessedDevIn.append(temp)
        count+=1
count = 0

for line3 in unprocessedDevOut:
    if len(line3)!=0 and line3!="\n":
        temp = line3.rstrip().split(" ")
        value , tag = temp[0], temp[1]
        ProcessedTrain.append([value,tag])
        count+=1

def process_all_data():
    return {
        "ProcessedTrain": ProcessedTrain,
        "ProcessedDevIn": ProcessedDevIn,
        "ProcessedDevOut": ProcessedDevOut
    }

