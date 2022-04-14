import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def load_word_embeddings(filename, word2Idx):
    # Loading glove embeddings (50d vectors)
    embeddings_index = {}
    f = open(filename, encoding="utf-8")
    for line in f:
        values = line.strip().split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # representing the word
        embeddings_index[word] = coefs
    f.close()

    EMBEDDING_VECTOR_SIZE = 50
    embedding_matrix = np.zeros((len(word2Idx), EMBEDDING_VECTOR_SIZE))
    # Word embeddings for the tokens
    for word,i in word2Idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_emb_layer(weights_matrix, trainable=True):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.weight.data = torch.from_numpy(weights_matrix)
    if trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers, n_classes):
        super().__init__()
        self.embedding, _, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilistm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, n_classes)

    def forward(self, inp):
        h0 = torch.zeros(self.num_layers*2, inp.size(0), self.hidden_size)#.to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, inp.size(0), self.hidden_size)#.to(device)
        out, _ = self.bilistm(self.embedding(inp), (h0, c0))
        return F.log_softmax(self.linear(out), dim=-1)

def test(model, test_loader, epoch, epochs, status):
    model.eval()
    list_y_true, list_y_pred = [], []
    with torch.no_grad():
            test_accuracy, test_f1_macro, test_f1_micro = 0, 0, 0
            for word, label in test_loader:
                output = model(word)
                argmaxed = torch.argmax(output, dim=-1)
                mask = label > -1
                relevant = argmaxed[mask]
                y_pred = relevant
                y_true = label[mask]
                list_y_pred.extend(y_pred)
                list_y_true.extend(y_true)
                test_accuracy += ((relevant == label[mask]).sum().item() / len(relevant))
                test_f1_macro += f1_score(y_true, y_pred, average='macro')
                test_f1_micro += f1_score(y_true, y_pred, average='micro')
            print("Epoch: {}/{},".format(epoch+1, epochs) if status == "Validation" else "",
                "{} Accuracy: {:.3f},".format(status, test_accuracy/len(test_loader)),
                "{} F1-macro: {:.3f},".format(status, test_f1_macro/len(test_loader)),
                "{} F1-micro: {:.3f},".format(status, test_f1_micro/len(test_loader)))
    return list_y_true, list_y_pred
