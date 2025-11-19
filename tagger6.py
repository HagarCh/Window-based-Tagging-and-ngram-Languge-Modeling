#%%
import torch
import numpy
from utils import get_data_path
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import itertools
import time
import random

SEED = 42  # Any fixed number
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load and clean the corpus 
def load_corpus(path):
    with open(path, "r") as f:
        line = f.read()
    
    text = line.lower()
    return text

def build_vocab(text):
    # Get unique characters in the text
    vocab = sorted(list(set(text)))

    # Create char-to-index and index-to-char mappings
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Vocabulary size: {len(vocab)} characters")
    return vocab, char_to_idx, idx_to_char


def create_datasets(text, k, val_ratio=0.1):
    # Split into training and validation text
    split_idx = int(len(text) * (1 - val_ratio))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"Train chars: {len(train_text):,}, Val chars: {len(val_text):,}")

    def make_examples(text):
        X, Y = [], []
        for i in range(len(text) - k):
            context = text[i:i + k]
            target = text[i + k]
            X.append(context)
            Y.append(target)
        return X, Y

    X_train, Y_train = make_examples(train_text)
    X_val, Y_val = make_examples(val_text)

    print(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
    return X_train, Y_train, X_val, Y_val

class NgramModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes, dropout_rate, k, embedding_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.window_size = k
        self.input_dim = k * embedding_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        flattened = embedded.view(tokens.size(0),
                                  -1)  # because we have batch_size x win_size x embedding_size, tokens.size(0) is the batch size
        hidden = torch.tanh(self.fc1(flattened))
        output =self.fc2(hidden)
        return output

class ConvertDatasetToTorch(torch.utils.data.Dataset):
    def __init__(self, X, Y, char_to_idx):
        """
        X: list of strings, each of length k (context window)
        Y: list of single characters (next character)
        char_to_idx: dictionary mapping characters to integer indices
        """
        self.X = X
        self.Y = Y
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert input sequence (k chars) to list of indices
        x_str = self.X[idx]
        y_char = self.Y[idx]

        x_indices = torch.tensor([self.char_to_idx[ch] for ch in x_str], dtype=torch.long)
        y_index = torch.tensor(self.char_to_idx[y_char], dtype=torch.long)

        return x_indices, y_index

def train_model(model, dataLoader, criterion, optimizer):
    model = model.to(device)
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


def evaluate_loader(model, data_loader, criterion):
    """
    Evaluates the model's accuracy on a dataset.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for tokens, labels in data_loader:
            tokens = tokens.to(device)  # move to device
            labels = labels.to(device)
            outputs = model(tokens)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # sum over batch

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0

    return accuracy, avg_loss

def emphsample(model, prefix, num_chars, char_to_idx, idx_to_char, k, temperature=1.0):
    model.eval()
    generated = prefix

    for _ in range(num_chars):
        # Get the last k characters from the generated text
        context = generated[-k:]
        if len(context) < k:
            context = ' ' * (k - len(context)) + context  # pad if too short

        # Convert context to tensor of indices
        context_indices = [char_to_idx.get(ch, char_to_idx[' ']) for ch in context]
        input_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, k)

        with torch.no_grad():
            logits = model(input_tensor)  # shape: (1, vocab_size)

        # Apply temperature scaling
        logits = logits.squeeze() / temperature
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Sample next character from the probability distribution
        next_char_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_char_idx]

        # Append sampled character to the sequence
        generated += next_char

    return generated

def run_grid_search(text, char_to_idx, idx_to_char, device):
    k_values = [1, 3, 5, 10]
    embedding_dims = [32, 64]
    hidden_dims = [128, 256]
    learning_rates = [1e-3, 5e-4]
    dropout_rate = 0.3
    batch_size = 64
    num_epochs = 10

    results = []

    for k, emb_dim, hid_dim, lr in itertools.product(k_values, embedding_dims, hidden_dims, learning_rates):
        print(f"\n🔧 Testing config: k={k}, emb_dim={emb_dim}, hid_dim={hid_dim}, lr={lr}")

        # Prepare data
        X_train, Y_train, X_val, Y_val = create_datasets(text, k)
        train_dataset = ConvertDatasetToTorch(X_train, Y_train, char_to_idx)
        val_dataset = ConvertDatasetToTorch(X_val, Y_val, char_to_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model, loss, optimizer
        vocab_size = len(char_to_idx)
        model = NgramModel(vocab_size, hid_dim, vocab_size, dropout_rate, k, embedding_dim=emb_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train and evaluate
        train_losses, val_losses, val_accuracies = [], [], []
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            val_acc, val_loss = evaluate_loader(model, val_loader, criterion)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"  Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")


        
        # Save best model so far (lowest val loss)
        if len(results) == 0 or val_loss < results[0]['val_loss']:
            torch.save(model.state_dict(), f"best_model_k{k}_emb{emb_dim}_hid{hid_dim}_lr{lr}.pt")
            print(f"📦 Saved best model so far: k={k}, emb={emb_dim}, hid={hid_dim}, lr={lr}, val_loss={val_loss:.4f}")

        # Record result
        results.append({
            'k': k,
            'embedding_dim': emb_dim,
            'hidden_dim': hid_dim,
            'lr': lr,
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 4)
        })


    # Sort by best validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    print("\n Grid Search Results (sorted by val_loss):")
    for res in results:
        print(res)


if __name__ == "__main__":
    path = get_data_path("input.txt", "lm-data/eng-data")
    text = load_corpus(path)
    vocab, char_to_idx, idx_to_char = build_vocab(text)
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # run_grid_search(text, char_to_idx, idx_to_char, device)


    for k in [1, 3, 5, 10]:
        print(f"Training model with k={k}")
        X_train, Y_train, X_val, Y_val = create_datasets(text, k)
        train_dataset = ConvertDatasetToTorch(X_train, Y_train, char_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataset = ConvertDatasetToTorch(X_val, Y_val, char_to_idx)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # === Model, Loss, Optimizer ===
        epochs = 15
        dropout_rate = 0.3
        hidden_dim=256
        num_classes = len(char_to_idx)
        vocab_size = len(char_to_idx)
        model = NgramModel(vocab_size, hidden_dim, num_classes,
                                dropout_rate, k, embedding_dim=64).to(device)
            
        optimizer= optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        train_losses, dev_losses, dev_acc = [], [], []
        total_start = time.time()

        # === Training Loop ===
        for epoch in range(epochs):
            epoch_start = time.time()
            loss = train_model(model, train_loader, criterion, optimizer)
            acc, dev_loss = evaluate_loader(model, val_loader, criterion)
            train_losses.append(loss)
            dev_losses.append(dev_loss)
            dev_acc.append(acc)
            print(f"  Epoch {epoch+1}/{epochs} | Train: {loss:.4f} | Val: {dev_loss:.4f} | Acc: {acc:.4f} | Time: {time.time() - epoch_start:.2f}s")
        
        print(f"✅ Done with k = {k} | Best Val Loss: {min(dev_losses):.4f} | Best Acc: {max(dev_acc):.4f} | Total Time: {(time.time() - total_start)/60:.2f} minutes")
        # Save the trained model to a file
        torch.save(model.state_dict(), f"trained_model_k{k}_emb64_hid{hidden_dim}_lr0005.pt")
        print("✅ Model saved.")


#%%
import torch
import numpy
from utils import get_data_path
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import itertools
import time
import random

def emphsample(model, prefix, num_chars, char_to_idx, idx_to_char, k, temperature=1.0):
    model.eval()
    generated = prefix

    for _ in range(num_chars):
        # Get the last k characters from the generated text
        context = generated[-k:]
        if len(context) < k:
            context = ' ' * (k - len(context)) + context  # pad if too short

        # Convert context to tensor of indices
        context_indices = [char_to_idx.get(ch, char_to_idx[' ']) for ch in context]
        input_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, k)

        with torch.no_grad():
            logits = model(input_tensor)  # shape: (1, vocab_size)

        # Apply temperature scaling
        logits = logits.squeeze() / temperature
        probs = torch.softmax(logits, dim=0).cpu().numpy()

        # Sample next character from the probability distribution
        next_char_idx = np.random.choice(len(probs), p=probs)
        next_char = idx_to_char[next_char_idx]

        # Append sampled character to the sequence
        generated += next_char

    return generated



class NgramModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_classes, dropout_rate, k, embedding_dim=50):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.window_size = k
        self.input_dim = k * embedding_dim
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, tokens):
        embedded = self.embedding(tokens)
        flattened = embedded.view(tokens.size(0),
                                  -1)  # because we have batch_size x win_size x embedding_size, tokens.size(0) is the batch size
        hidden = torch.tanh(self.fc1(flattened))
        output =self.fc2(hidden)
        return output
    
def load_corpus(path):
    with open(path, "r") as f:
        line = f.read()
    
    text = line.lower()
    return text
def build_vocab(text):
    # Get unique characters in the text
    vocab = sorted(list(set(text)))

    # Create char-to-index and index-to-char mappings
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    print(f"Vocabulary size: {len(vocab)} characters")
    return vocab, char_to_idx, idx_to_char


# Reconstruct the model architecture with the same params
global device
path = get_data_path("input.txt", "lm-data/eng-data")
text = load_corpus(path)
vocab, char_to_idx, idx_to_char = build_vocab(text)
epochs = 15
dropout_rate = 0.3
hidden_dim=256
num_classes = len(char_to_idx)
vocab_size = len(char_to_idx)
k=5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NgramModel(vocab_size, hidden_dim, num_classes,
                     dropout_rate, k, embedding_dim=64).to(device)

    # Load weights
model.load_state_dict(torch.load("/home/dsi/hagarch/ass2/trained_model_k5_emb64_hid256_lr0005.pt"))
model.eval()  # set to evaluation mode
num_chars=100
prefix="To be"
print(emphsample(model, prefix, num_chars,
char_to_idx, idx_to_char, k, temperature=0.8))