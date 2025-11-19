# %%
import math

from utils import (arrange_data_and_chars, build_label_vocab, build_vocab, build_vocab_chars, word2idx_embed_vocab, load_pretrained_embeddings)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import numpy as np
import argparse
from torch.nn.utils.rnn import pad_sequence


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, num_filters, filter_size, output_dim):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        b = math.sqrt(3/char_emb_dim)
        nn.init.uniform_(self.char_embedding.weight, -b, b)

        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=0
        )
        self.output_dim = output_dim
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, word_char_indices):
        x = self.char_embedding(word_char_indices)  # (B*W, max_len, emb_dim)
        x = x.permute(0, 2, 1)  # to (B*W, emb_dim, max_len)
        x = torch.relu(self.conv(x))
        x, _ = torch.max(x, dim=2)
        return self.fc(x)


class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, chars_vocab_size, hidden_dim, num_classes, dropout_rate, use_pretrained, window_size=5, embedding_dim=50, char_embedding_dim=30):
        super().__init__()
        if use_pretrained:
            # vocab = load_vocab("vocab.txt")
            pretrained_matrix = load_pretrained_embeddings("wordVectors.txt")
            self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)


        self.char_cnn = CharCNN(char_vocab_size=chars_vocab_size, char_emb_dim=char_embedding_dim,
                                num_filters=30, filter_size=3, output_dim=30)


        self.window_size = window_size
        self.input_dim = window_size * embedding_dim

        self.fc1 = nn.Linear(window_size * (embedding_dim+char_embedding_dim), hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens, chars_tokens):
        batch_size = tokens.size(0)
        word_embedded = self.embedding(tokens)

        batch_size, window_size, max_word_len = chars_tokens.size()
        chars_tokens_flat = chars_tokens.view(-1, max_word_len)
        char_embedded_flat = self.char_cnn(chars_tokens_flat)
        char_embedded = char_embedded_flat.view(batch_size, window_size, -1)

        combined_embedded = torch.cat([word_embedded, char_embedded], dim=2)

        flattened = combined_embedded.view(tokens.size(0), -1)
        hidden = torch.tanh(self.fc1(flattened))
        output = self.fc2(hidden)
        return output

class ConvertDatasetToTorch(torch.utils.data.Dataset):
    def __init__(self, data, chars_data, word2idx, char2idx, label2idx=None, use_pretrained=False, has_labels=True):
        self.data = data
        self.chars_data = chars_data
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.label2idx = label2idx
        self.has_labels = has_labels
        self.use_pretrained = use_pretrained


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.has_labels:
            window_words, label = self.data[idx]
            window_chars, label = self.chars_data[idx]
        else:
            window_words = self.data[idx]
            window_chars = self.chars_data[idx]

        if self.use_pretrained:
            # Lowercase only at lookup
            word_indices = [self.word2idx.get(
                (w.lower() if isinstance(w, str) else w[0].lower()), 
                self.word2idx["<unk>"]
            ) for w in window_words]
        else:
            word_indices = [self.word2idx.get(
                (w if isinstance(w, str) else w[0]), 
                self.word2idx["<UNK>"]
            ) for w in window_words]

        chars_indices = []
        for word in window_chars:
            word = word[0]
            chars_indices.append([self.char2idx.get(
                (c if isinstance(c, str) else c[0]),
                self.char2idx["<UNK>"]
            ) for c in word])

        MAX_CHAR_LEN = 20

        def pad_or_truncate(seq, max_len=MAX_CHAR_LEN, pad_value=0):
            if len(seq) < max_len:
                return seq + [pad_value] * (max_len - len(seq))
            else:
                return seq[:max_len]

        padded = [pad_or_truncate(seq) for seq in chars_indices]
        padded_chars_tensor = torch.tensor(padded, dtype=torch.long)

        if self.has_labels:
            label_index = self.label2idx[label]
            return (torch.tensor(word_indices, dtype=torch.long), padded_chars_tensor,
                    torch.tensor(label_index, dtype=torch.long))
        else:
            realword = window_words[2]
            return (torch.tensor(word_indices, dtype=torch.long), padded_chars_tensor,
                    realword)



def train_model(model, dataLoader, criterion, optimizer):
    model.train()
    total_loss = 0
    total = 0

    for token, chars_tokens, label in dataLoader:
        token = token.to(device)  # move to device
        chars_tokens = chars_tokens.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(token, chars_tokens)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        batch_size = label.size(0)
        total_loss += loss.item() * batch_size  # Sum up the total loss
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
    for tokens, chars_tokens, labels in data_loader:
        tokens = tokens.to(device)  # move to device
        chars_tokens = chars_tokens.to(device)
        labels = labels.to(device)
        outputs = model(tokens, chars_tokens)
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
            for tokens, chars_tokens, words_batch in test_loader:
                tokens = tokens.to(device)
                chars_tokens = chars_tokens.to(device)
                outputs = model(tokens, chars_tokens)
                preds = torch.argmax(outputs, dim=1)
                preds = preds.cpu().numpy()

                for word, pred_idx in zip(words_batch, preds):
                    label = idx2label[pred_idx]
                    if word == "<pad>" or word == "<PAD>":
                        continue
                    f.write(f"{word}    {label}\n")
    print(f"Test predictions written to {output_file}")



def train_and_eval_ner(use_pretrained):
    # === Setup data for NER ===
    dataset = arrange_data_and_chars(use_pretrained)
    dataset_train, chars_train = dataset["ner"]["train"]
    dataset_dev, chars_dev = dataset["ner"]["dev"]
    dataset_test, chars_test = dataset["ner"]["test"]

    classes_ner_train = build_label_vocab(dataset_train)
    if use_pretrained:
        vocab_ner_train = word2idx_embed_vocab("vocab.txt")  # pre-aligned with embeddings
    else:
        vocab_ner_train = build_vocab(dataset_train)

    chars_vocab_ner_train = build_vocab_chars(chars_train)


    train_dataset_ner = ConvertDatasetToTorch(dataset_train, chars_train, vocab_ner_train, chars_vocab_ner_train, classes_ner_train, use_pretrained, has_labels=True)
    dev_dataset_ner = ConvertDatasetToTorch(dataset_dev, chars_dev, vocab_ner_train, chars_vocab_ner_train, classes_ner_train, use_pretrained, has_labels=True)

    train_loader_ner = DataLoader(train_dataset_ner, batch_size=16, shuffle=True)
    dev_loader_ner = DataLoader(dev_dataset_ner, batch_size=16, shuffle=True)


    # === Model, Loss, Optimizer ===
    epochs_ner = 20
    dropout_rate_ner = 0.5
    hidden_dim_ner=128
    num_classes_ner=len(classes_ner_train)
    vocab_size_ner=len(vocab_ner_train)
    chars_vocab_size_ner=len(chars_vocab_ner_train)
    model_ner = SequenceTagger(vocab_size_ner, chars_vocab_size_ner, hidden_dim_ner, num_classes_ner,
                            dropout_rate_ner,use_pretrained, window_size=5, embedding_dim=50).to(device)
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
    test_dataset_ner = ConvertDatasetToTorch(dataset_test, chars_test, vocab_ner_train, chars_vocab_ner_train, classes_ner_train, use_pretrained, has_labels=False)
    test_loader_ner = DataLoader(test_dataset_ner, batch_size=16)

    idx2label_ner = {v: k for k, v in classes_ner_train.items()}

    predict_and_save_test(model_ner, test_loader_ner, idx2label_ner,  "test5.ner")


    
    # save the model
    torch.save(model_ner.state_dict(), "model_ner.pt")
    np.save("ner_acc_without_O.npy", np.array(dev_accuracies_ner_without_O))
    np.save("ner_acc.npy", np.array(dev_accuracies_ner)) 
    np.save("ner_losses_dev.npy", np.array(dev_losses_ner))
    np.save("ner_losses_train.npy", np.array(train_losses_ner)) 


def train_and_eval_pos(use_pretrained):
    # === Setup data for POS ===
    dataset = arrange_data_and_chars(use_pretrained)
    dataset_train, chars_train = dataset["pos"]["train"]
    dataset_dev, chars_dev = dataset["pos"]["dev"]
    dataset_test, chars_test = dataset["pos"]["test"]

    classes_pos_train = build_label_vocab(dataset_train)
    if use_pretrained:
        vocab_pos_train = word2idx_embed_vocab("vocab.txt")  # pre-aligned with embeddings
    else:
        vocab_pos_train = build_vocab(dataset_train)

    chars_vocab_pos_train = build_vocab_chars(chars_train)


    train_dataset_pos = ConvertDatasetToTorch(dataset_train, chars_train, vocab_pos_train, chars_vocab_pos_train, classes_pos_train, use_pretrained, has_labels=True)
    dev_dataset_pos = ConvertDatasetToTorch(dataset_dev, chars_dev, vocab_pos_train, chars_vocab_pos_train, classes_pos_train, use_pretrained, has_labels=True)

    train_loader_pos = DataLoader(train_dataset_pos, batch_size=64, shuffle=True)
    dev_loader_pos = DataLoader(dev_dataset_pos, batch_size=64, shuffle=True)


    # === Model, Loss, Optimizer ===
    epochs_pos = 7
    dropout_rate_pos = 0.2
    hidden_dim_pos=256
    num_classes_pos=len(classes_pos_train)
    vocab_size_pos=len(vocab_pos_train)
    chars_vocab_size_pos=len(chars_vocab_pos_train)
    model_pos = SequenceTagger(vocab_size_pos, chars_vocab_size_pos, hidden_dim_pos, num_classes_pos,
                            dropout_rate_pos,use_pretrained, window_size=5, embedding_dim=50).to(device)

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

    test_dataset_pos = ConvertDatasetToTorch(dataset_test, chars_test, vocab_pos_train, chars_vocab_pos_train, classes_pos_train, use_pretrained, has_labels=False)
    test_loader_pos = DataLoader(test_dataset_pos, batch_size=64)
    idx2label_pos = {v: k for k, v in classes_pos_train.items()}

    predict_and_save_test(model_pos, test_loader_pos, idx2label_pos, "test5.pos")

    torch.save(model_pos.state_dict(), "model_pos.pt")
    np.save("pos_acc.npy", np.array(dev_accuracies))
    np.save("pos_losses_dev.npy", np.array(dev_losses)) 
    np.save("pos_losses_train.npy", np.array(train_losses))

if __name__ == "__main__":
    global device
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

