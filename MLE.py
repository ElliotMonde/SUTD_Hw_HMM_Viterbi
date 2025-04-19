from Processed_Data import process_all_data
data = process_all_data()
train_data = data["ProcessedTrain"]
dev_in_data = data["ProcessedDevIn"]
dev_out_data = data["ProcessedDevOut"]

def mle(data , k = 0):
    countYX = {}
    countY = {}
    for word,tag in train_data:
        countY[tag] = countY.get(tag, 0) + 1
        if tag not in countYX:
            countYX[tag] = {}
        countYX[tag][word] = countYX[tag].get(word, 0) + 1
    emissionProbs = {}
    for tag in countYX:
        emissionProbs[tag] = {}
        for word in countYX[tag]:
            emissionProbs[tag][word] = countYX[tag][word] / countY[tag]
    return emissionProbs

def smoothenedMLE(data, k = 3):
    wordCounts = {}
    for word, tag in data:
        wordCounts[word] = wordCounts.get(word, 0) + 1
    countYX = {}
    countY = {}
    for word, tag in data:
        processedWord = word if wordCounts[word] >= k else "#UNK#"
        countY[tag] = countY.get(tag, 0) + 1
        if tag not in countYX:
            countYX[tag] = {}
        countYX[tag][processedWord] = countYX[tag].get(processedWord, 0) + 1
    emissionProbs = {}
    for tag in countYX:
        emissionProbs[tag] = {}
        for word in countYX[tag]:
            emissionProbs[tag][word] = countYX[tag][word] / countY[tag]
    for tag in countY:
        if "#UNK#" not in emissionProbs.get(tag, {}):
            emissionProbs[tag]["#UNK#"] = countYX.get(tag, {}).get("#UNK#", 0) / countY[tag]
    return emissionProbs

print(smoothenedMLE(train_data))