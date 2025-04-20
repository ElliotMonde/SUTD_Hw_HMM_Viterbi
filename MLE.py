from Processed_Data import process_all_data
data = process_all_data()
train_data = data["ProcessedTrain"]
dev_in_data = data["ProcessedDevIn"]
dev_out_data = data["ProcessedDevOut"]

def train_mle(data , k = 0):
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

def train_smoothenedMLE(data, k = 3):
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

def predict_tags(emission_probs, input_data, output_file):
    vocabulary = []
    for tag in emission_probs:
        if tag not in vocabulary:
            vocabulary.append(emission_probs[tag].keys())
    with open(output_file, 'w') as fout:
        count = 0
        for line in input_data:
            word = line[0]
            processed_word = word if word in vocabulary else "#UNK#"
            bestTag = max(emission_probs.keys(),key=lambda tag: emission_probs[tag].get(processed_word, 0))
            fout.write(f"{word} {bestTag}\n")
            fout.write("\n")
print(predict_tags(train_smoothenedMLE(train_data),dev_in_data,"EN/EN/dev.p2.out"))