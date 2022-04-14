import argparse

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

import embedding_model as em
import preprocessing as pre


def compute_class_weights(split_train):
    class_frequency = {}
    for labeled_text in split_train:
        for _, label in labeled_text:
            if label not in class_frequency:
                class_frequency[label] = 0
            class_frequency[label] += 1

    summed = sum(class_frequency.values())
    weights = torch.tensor(list(class_frequency.values())) /  summed
    weights = 1./weights
    return weights


def main(args):
    # load data
    split_train = pre.split_text("data/train.conll")
    split_dev = pre.split_text("data/dev.conll")
    split_test = pre.split_text("data/test.conll")
    # forming vocabulary
    labelSet = set()
    wordSet = set()
    # words and labels
    for data in [split_train]:
        for labeled_text in data:
            for word, label in labeled_text:
                labelSet.add(label)
                wordSet.add(word.lower())

    # mapping words and labels to indices
    # Sort the set to ensure 'O' is assigned to 0
    sorted_labels = sorted(list(labelSet), key=len)
    # Create mapping for labels
    label2Idx = {}
    for label in sorted_labels:
        label2Idx[label] = len(label2Idx)
    idx2Label = {v: k for k, v in label2Idx.items()}

    # defining padding and unknown tokens
    word2Idx = {}
    word2Idx["PADDING_TOKEN"] = -1
    word2Idx["UNKNOWN_TOKEN"] = 0
    for word in wordSet:
        word2Idx[word] = len(word2Idx)

    # forming sentences and labels
    train_sentences, train_labels = pre.createMatrices(split_train, word2Idx, label2Idx)
    valid_sentences, valid_labels = pre.createMatrices(split_dev, word2Idx, label2Idx)
    test_sentences, test_labels = pre.createMatrices(split_test, word2Idx, label2Idx)

    # padding sequences
    MAX_LENGTH = max(list(map(len, train_sentences))) + 1
    train_features, train_labels  = pre.pad_seq(train_sentences, MAX_LENGTH), pre.pad_seq(train_labels, MAX_LENGTH, label=True)
    valid_features, valid_labels = pre.pad_seq(valid_sentences, MAX_LENGTH), pre.pad_seq(valid_labels, MAX_LENGTH, label=True)
    test_features, test_labels = pre.pad_seq(test_sentences, MAX_LENGTH), pre.pad_seq(test_labels, MAX_LENGTH, label=True)

    # model definition
    hidden_layer_dim, num_of_hidden_layer = 100, 1
    n_classes = len(label2Idx)
    embedding_matrix = em.load_word_embeddings("glove.6B.50d.txt", word2Idx)
    model = em.ToyNN(embedding_matrix, hidden_layer_dim, num_of_hidden_layer, n_classes).float()

    # model hyperparameters
    batch_size, lr = args.batch_size, 0.01
    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create dataloaders
    train_data = TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels))
    valid_data = TensorDataset(torch.from_numpy(valid_features), torch.from_numpy(valid_labels))
    test_data = TensorDataset(torch.from_numpy(test_features), torch.from_numpy(test_labels))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # compute class weights
    class_weights = compute_class_weights(split_train)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights, reduction='mean')

    for epoch in range(epochs):
        # training
        model.train()
        train_accuracy = 0
        for word, label in train_loader:
            # forward propagation
            output = model(word)
            loss = criterion(output.view(-1, output.shape[-1]), label.view(-1))
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            argmaxed = torch.argmax(output, dim=-1)
            mask = label > -1
            relevant = argmaxed[mask]
            train_accuracy += ((relevant == label[mask]).sum().item() / len(relevant))
        print("Epoch: {}/{},".format(epoch+1, epochs),
                "Train Accuracy: {:.3f},".format(train_accuracy/len(train_loader)))

        # validation
        _, _ = em.test(model, valid_loader, epoch, epochs, "Validation")

    # test/evaluation
    y_test, y_pred = em.test(model, test_loader, None, None, "Test")

    # classification report
    print(classification_report(y_test, y_pred, target_names=list(label2Idx.keys())))

    # sample sentence, and why our model fails
    sample_sentence = [[["John", "B-PER"], ["lives", "O"], ["in", "O"], ["New", "B-LOC"], ["York", "I-LOC"]]]
    features, labels = pre.createMatrices(sample_sentence, word2Idx, label2Idx)
    features_pad, labels_pad = pre.pad_seq(features, MAX_LENGTH), pre.pad_seq(labels, MAX_LENGTH, label=True)
    sample_dataset = TensorDataset(torch.from_numpy(features_pad), torch.from_numpy(labels_pad))
    loader = DataLoader(sample_dataset, shuffle=False, batch_size=32)
    model.eval()
    with torch.no_grad():
            test_accuracy = 0
            for word, label in loader:
                output = model(word)
                argmaxed = torch.argmax(output, dim=-1)
                mask = label > -1
                relevant = argmaxed[mask]
                y_pred = relevant
                y_true = label[mask]
                test_accuracy += ((relevant == label[mask]).sum().item() / len(relevant))
            print(30*"-")
            print("Predicted:", list(map(lambda f: idx2Label[f], y_pred.detach().numpy())))
            print("True:     ", list(map(lambda f: idx2Label[f], y_true.detach().numpy())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="batch-size")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="Batch Size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        required=False,
        help="Epochs"
    )
    args = parser.parse_args()
    main(args)
