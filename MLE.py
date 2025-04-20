from Processed_Data import process_all_data
data = process_all_data()
train_data = data["ProcessedTrain"]
dev_in_data = data["ProcessedDevIn"]
dev_out_data = data["ProcessedDevOut"]

def emission_mle(data):
    countYX = {}
    countY = {}
    for word,tag in data:
        if tag == "START": 
            continue
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

def transition_mle(data):
    tags = {}
    count = {}

    START_S = "START"
    STOP_S = "STOP"

    prev_state = START_S

    for _, curr_state in data:
        if not (curr_state == START_S and prev_state == START_S):
            if curr_state != START_S: 
                tags[(prev_state, curr_state)] = tags.get((prev_state, curr_state), 0) + 1
            else:
                tags[(prev_state, STOP_S)] = tags.get((prev_state, STOP_S), 0) + 1
            count[curr_state] = count.get(curr_state, 0) + 1
       
        prev_state = curr_state
    
    for transition in tags:
            tags[transition] /= count[transition[0]]
    
    return tags
    
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

print(predict_tags(smoothenedMLE(train_data),dev_in_data,"EN/EN/dev.p2.out"))