from utils import (arrange_data, build_label_vocab, build_vocab, word2idx_embed_vocab, load_pretrained_embeddings)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import argparse
from itertools import product


SEED = 42  # Any fixed number
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes, dropout_rate, use_pretrained, window_size=5, embedding_dim=50):
        super().__init__()
        if use_pretrained:
            pretrained_matrix = load_pretrained_embeddings("wordVectors.txt")
            self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False) 
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.window_size = window_size
        self.input_dim = window_size * embedding_dim

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # <<< Add this line
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        flattened = embedded.view(tokens.size(0),
                                  -1)  # because we have batch_size x win_size x embedding_size, tokens.size(0) is the batch size
        hidden = self.dropout(torch.tanh(self.fc1(flattened)))
        output = self.fc2(hidden)
        return output
    


class ConvertDatasetToTorch(torch.utils.data.Dataset):
    def __init__(self, data, word2idx, label2idx=None, use_pretrained = False, has_labels=True):
        self.data = data
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.has_labels = has_labels
        self.use_pretrained = use_pretrained


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.has_labels:
            window_words, label = self.data[idx]
        else:
            window_words = self.data[idx]
        
        if self.use_pretrained:
            word_indices = [
                self.word2idx.get(
                    (w.lower() if isinstance(w, str) else w[0].lower()),
                    self.word2idx["<unk>"]
                )
                for w in window_words
                ]
        else:
             word_indices = [self.word2idx.get(
                (w if isinstance(w, str) else w[0]), 
                self.word2idx["<UNK>"]
                ) for w in window_words]

        if self.has_labels:
            label_index = self.label2idx[label]
            return torch.tensor(word_indices, dtype=torch.long), torch.tensor(label_index, dtype=torch.long)
        else:
            realword = window_words[2]  # center word
            return torch.tensor(word_indices, dtype=torch.long), realword

def train_model(model, dataLoader, criterion, optimizer):
    model.train()  # puts the model in training mode (for batchnorm and dropout)
    total_loss = 0
    total = 0  # Track number of samples

    for token, label in dataLoader:
        token = token.to(device)  # move to device
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(token)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size  # Sum up the total loss (not mean of means)
        total += batch_size

    return total_loss / total


def evaluate_loader(model, data_loader, label_vocab, criterion, is_ner_task=False):
    """
    Evaluates the model's accuracy on a dataset.
    """
    # model.eval()
    correct = 0
    total = 0
    ner_correct = 0
    ner_total = 0
    total_loss = 0.0
   
    # Only look for 'O' if this is an NER task
    o_label_idx = None
    if is_ner_task:
        o_label_idx = label_vocab['O']

    # with torch.no_grad():
    for tokens, labels in data_loader:
        tokens = tokens.to(device)  # move to device
        labels = labels.to(device)
        outputs = model(tokens)
        preds = torch.argmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)  # sum over batch

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # If NER, compute accuracy on non-'O' labels (true entities)
        if is_ner_task and o_label_idx is not None:
            mask = labels != o_label_idx
            ner_correct += ((preds == labels) & mask).sum().item()
            ner_total += mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    if is_ner_task:
        ner_accuracy = (ner_correct / ner_total) if is_ner_task and ner_total > 0 else None
    else:
        ner_accuracy = None
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss, ner_accuracy

def predict_and_save_test(model, test_loader, idx2label, output_file):
    model.eval()

    with open(output_file, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for tokens, words_batch in test_loader:
                tokens = tokens.to(device)
                outputs = model(tokens)
                preds = torch.argmax(outputs, dim=1)
                preds = preds.cpu().numpy()

                for word, pred_idx in zip(words_batch, preds):
                    label = idx2label[pred_idx]
                    if word == "<pad>":
                        continue
                    f.write(f"{word}    {label}\n")
    print(f"Test predictions written to {output_file}")



def train_and_eval_ner(use_pretrained):
    # === Setup data for NER ===
    dataset = arrange_data(use_pretrained)
    classes_ner_train = build_label_vocab(dataset["ner"]["train"])
    if use_pretrained:
        vocab_ner_train = word2idx_embed_vocab("vocab.txt")  # pre-aligned with embeddings
    else:
        vocab_ner_train = build_vocab(dataset["ner"]["train"])
    train_dataset_ner = ConvertDatasetToTorch(dataset["ner"]["train"], vocab_ner_train, classes_ner_train, use_pretrained, has_labels=True)
    dev_dataset_ner = ConvertDatasetToTorch(dataset["ner"]["dev"], vocab_ner_train, classes_ner_train, use_pretrained, has_labels=True)
    train_loader_ner = DataLoader(train_dataset_ner, batch_size=16, shuffle=True)
    dev_loader_ner = DataLoader(dev_dataset_ner, batch_size=16, shuffle=True)

    # === Model, Loss, Optimizer ===
    if use_pretrained:
        epochs_ner = 4
        dropout_rate_ner = 0.0
        hidden_dim_ner=128

    else:
        epochs_ner = 25
        dropout_rate_ner = 0.5
        hidden_dim_ner=128

    num_classes_ner=len(classes_ner_train)
    vocab_size_ner=len(vocab_ner_train)

    model_ner = SequenceTagger(vocab_size_ner, hidden_dim_ner, num_classes_ner,
                            dropout_rate_ner,use_pretrained, window_size=5, embedding_dim=50).to(device)
    
    if use_pretrained:
        optimizer_ner = optim.Adam(model_ner.parameters(), lr=0.0005, weight_decay=1e-5)
    else:
        optimizer_ner = optim.Adam(model_ner.parameters(), lr=0.0005, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    train_losses_ner, dev_losses_ner, dev_accuracies_ner, dev_accuracies_ner_without_O = [], [], [], []

    # === Training Loop ===
    for epoch in range(epochs_ner):
        loss = train_model(model_ner, train_loader_ner, criterion, optimizer_ner)
        acc, dev_loss, ner_acc = evaluate_loader(model_ner, dev_loader_ner, classes_ner_train, criterion, is_ner_task=True)
        train_losses_ner.append(loss)
        dev_losses_ner.append(dev_loss)
        dev_accuracies_ner.append(acc)
        dev_accuracies_ner_without_O.append(ner_acc)
        print(f"Epoch (Ner) {epoch + 1}/{epochs_ner} | Loss (train): {loss:.4f} | Loss (dev): {dev_loss:.4f} | NER Acc: {acc:.4f} | NER Acc (without O): {ner_acc:.4f}")
    
    # predict on the test set
    test_dataset_ner = ConvertDatasetToTorch(dataset["ner"]["test"], vocab_ner_train, classes_ner_train, use_pretrained, has_labels=False)
    test_loader_ner = DataLoader(test_dataset_ner, batch_size=32)
    idx2label_ner = {v: k for k, v in classes_ner_train.items()}
    if use_pretrained:
        test_res_file = "test3.ner"
    else:
        test_res_file = "test1.ner"
    predict_and_save_test(model_ner, test_loader_ner, idx2label_ner,  test_res_file)


    
    # save the model
    torch.save(model_ner.state_dict(), "model_ner.pt")
    np.save("ner_acc_without_O.npy", np.array(dev_accuracies_ner_without_O))
    np.save("ner_acc.npy", np.array(dev_accuracies_ner)) 
    np.save("ner_losses_dev.npy", np.array(dev_losses_ner))
    np.save("ner_losses_train.npy", np.array(train_losses_ner)) 


def train_and_eval_pos(use_pretrained):
    # === Setup data for POS ===
    dataset = arrange_data(use_pretrained)
    classes_pos_train = build_label_vocab(dataset["pos"]["train"])
    if use_pretrained:
        vocab_pos_train = word2idx_embed_vocab("vocab.txt")  # pre-aligned with embeddings
    else:
        vocab_pos_train = build_vocab(dataset["pos"]["train"])
    train_dataset_pos = ConvertDatasetToTorch(dataset["pos"]["train"], vocab_pos_train, classes_pos_train, use_pretrained, has_labels=True)
    dev_dataset_pos = ConvertDatasetToTorch(dataset["pos"]["dev"], vocab_pos_train, classes_pos_train, use_pretrained, has_labels=True)
    train_loader_pos = DataLoader(train_dataset_pos, batch_size=64, shuffle=True)
    dev_loader_pos = DataLoader(dev_dataset_pos, batch_size=64, shuffle=True)

    # === Model, Loss, Optimizer ===
    if use_pretrained:
        epochs_pos = 6
    else:
        epochs_pos = 8
    dropout_rate_pos = 0.0
    hidden_dim_pos=256
    num_classes_pos=len(classes_pos_train)
    vocab_size_pos=len(vocab_pos_train)
    model_pos = SequenceTagger(vocab_size_pos, hidden_dim_pos, num_classes_pos,
                            dropout_rate_pos, use_pretrained, window_size=5, embedding_dim=50).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_pos = optim.Adam(model_pos.parameters(), lr=0.0005, weight_decay=1e-5)

    train_losses, dev_losses, dev_accuracies = [], [], []
    # === Training Loop ===
    for epoch in range(epochs_pos):
        loss = train_model(model_pos, train_loader_pos, criterion, optimizer_pos)
        acc, dev_loss, _ = evaluate_loader(model_pos, dev_loader_pos, classes_pos_train, criterion, is_ner_task=False)
        train_losses.append(loss)
        dev_losses.append(dev_loss)
        dev_accuracies.append(acc)
        print(f"Epoch (Pos) {epoch + 1}/{epochs_pos} | Loss (train): {loss:.4f} | Loss (dev): {dev_losses[-1]:.4f} | Pos Acc: {dev_accuracies[-1]:.4f}")
    
    # predict on the test set
    test_dataset_pos = ConvertDatasetToTorch(dataset["pos"]["test"], vocab_pos_train, classes_pos_train, use_pretrained, has_labels=False)
    test_loader_pos = DataLoader(test_dataset_pos, batch_size=64)
    idx2label_pos = {v: k for k, v in classes_pos_train.items()}
    if use_pretrained:
        test_res_file = "test3.pos"
    else:
        test_res_file = "test1.pos"
    predict_and_save_test(model_pos, test_loader_pos, idx2label_pos, test_res_file)



    torch.save(model_pos.state_dict(), "model_pos.pt")
    np.save("pos_acc.npy", np.array(dev_accuracies))
    np.save("pos_losses_dev.npy", np.array(dev_losses)) 
    np.save("pos_losses_train.npy", np.array(train_losses))


def run_hyperparameter_search(task="pos", use_pretrained=False):
    dataset = arrange_data(use_pretrained)
    data_key = "pos" if task == "pos" else "ner"
    classes = build_label_vocab(dataset[data_key]["train"])
    vocab = word2idx_embed_vocab("vocab.txt") if use_pretrained else build_vocab(dataset[data_key]["train"])
    train_data = dataset[data_key]["train"]
    dev_data = dataset[data_key]["dev"]

    train_dataset = ConvertDatasetToTorch(train_data, vocab, classes, use_pretrained, has_labels=True)
    dev_dataset = ConvertDatasetToTorch(dev_data, vocab, classes, use_pretrained, has_labels=True)

    dropout_rates = [0.5]
    hidden_dims = [128, 256]
    learning_rates = [1e-3, 5e-4]
    batch_sizes = [64] if task == "pos" else [32]
    epochs = 20

    best_model = None
    best_val_acc = -1
    best_config = None

    for dropout, hidden_dim, lr, batch_size in product(dropout_rates, hidden_dims, learning_rates, batch_sizes):
        print(f"\n🔍 Testing config: dropout={dropout}, hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

        model = SequenceTagger(len(vocab), hidden_dim, len(classes), dropout, use_pretrained).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_model(model, train_loader, criterion, optimizer)

        acc, _, ner_acc = evaluate_loader(model, dev_loader, classes, criterion, is_ner_task=(task == "ner"))
        val_metric = ner_acc if task == "ner" else acc
        print(f"📊 Final Val Acc: {val_metric:.4f}")

        if val_metric > best_val_acc:
            best_val_acc = val_metric
            best_model = model
            best_config = (dropout, hidden_dim, lr, batch_size)

    print(f"\n✅ Best Config: Dropout={best_config[0]}, Hidden Dim={best_config[1]}, LR={best_config[2]}, Batch Size={best_config[3]} with Val Acc={best_val_acc:.4f}")
    torch.save(best_model.state_dict(), f"best_model_{task}.pt")

if __name__ == "__main__":
    global device
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['ner', 'pos'], required=True,
                        help="Which tagging task to run: ner or pos")
    parser.add_argument('--use_pretrained', action='store_true',
                        help="Use pre-trained word embeddings if set")
    args = parser.parse_args()

    if args.task == "ner":
        train_and_eval_ner(use_pretrained=args.use_pretrained)
    elif args.task == "pos":
        train_and_eval_pos(use_pretrained=args.use_pretrained)
    '''
    train_and_eval_ner(use_pretrained=False)
