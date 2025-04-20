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
        for line in input_data:
            word = line[0]
            processed_word = word if word in vocabulary else "#UNK#"
            bestTag = max(emission_probs.keys(),key=lambda tag: emission_probs[tag].get(processed_word, 0))
            fout.write(f"{word} {bestTag}\n")
            fout.write("\n")

print(predict_tags(smoothenedMLE(train_data),dev_in_data,"EN/dev.p2.out"))


def viterbi(seq, a, b, states):

    # init score matrix
    N = len(seq)
    T = len(states)
    score_m = [[0] * T for _ in range(N + 1)]
    backpointer = [[0] * T for _ in range(N + 1)]

    START_S = "START"
    STOP_S ="STOP"

    # init step 0 START
    for j in range(T):
        score_m[0][j] = a[(START_S, states[j])] * b[states[j], seq[0]]

    # forward
    for i in range(1, N):
        for j in range(T):
            parent = 0
            score = 0
            for k in range(0, T):
                score = score_m[i-1][k] * a[(states[k], states[j])] * b[states[j], seq[i]]
                if score_m[i][j] < score:
                    score_m[i][j] = score
                    parent = k
            backpointer[i][j] = parent
    
    # terminate recursion
    best_final_score = 0
    last_parent = None
    for j in range(T):        
        score_m[N][j] = score_m[N-1][j] * a[(states[j], STOP_S)]
        if score_m[N][j] > best_final_score:
            last_parent = j
            best_final_score = score_m[N][j]

    # backwards
    res = [None] * (N+1)
    res[N] = STOP_S
    res[N-1] = states[last_parent]

    for i in range(N-2, -1, -1):
        last_parent = backpointer[i+1][last_parent]
        res[i] = states[last_parent]

    return res
