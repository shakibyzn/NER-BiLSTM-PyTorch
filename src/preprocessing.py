import numpy as np


def split_text(filename):
    f = open(filename)
    sentence = []
    split_labeled_text = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence):
                split_labeled_text.append(sentence)
                sentence = []
            continue
        splits = line.split("\t")
        sentence.append([splits[0].strip(),splits[-1].strip()])
    if len(sentence):
        split_labeled_text.append(sentence)
        sentence = []
    return split_labeled_text


def createMatrices(data, word2Idx, label2Idx):
    sentences = []
    labels = []
    for split_labeled_text in data:
        wordIndices = []
        labelIndices = []
        for word, label in split_labeled_text:
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = word2Idx['UNKNOWN_TOKEN']
            wordIndices.append(wordIdx)
            labelIndices.append(label2Idx[label])
        sentences.append(wordIndices)
        labels.append(labelIndices)
    return sentences, labels


def pad_seq(sentences, seq_len, label=False):
    if not label:
        features = np.zeros((len(sentences), seq_len),dtype=int)
    else:
        features = -1 * np.ones((len(sentences), seq_len),dtype=int)

    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features
