import os
import torch
import random

WINDOW_SIZE = 2  # Context of two tokens before and after
def mask_10_percent(data, mask_ratio):
    masked_data = []
    for doc in data:
        masked_doc = []
        for token, label in doc:
            if random.random() < mask_ratio:
                masked_doc.append(("<unk>", label))
            else:
                masked_doc.append((token, label))
        masked_data.append(masked_doc)
    return masked_data

def load_documents(filename, use_pretrained):
    documents = []
    current_doc = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # New document detected
            if not line:
                if current_doc:
                    documents.append(current_doc)
                    current_doc = []
                continue

            parts = line.split()

            if len(parts) == 1 and filename.endswith("test"):
                token = parts[0]
                current_doc.append(token)
            elif len(parts) == 2:
                token, label = parts
                current_doc.append((token, label))

        # Append the last document if not empty
        if current_doc:
            documents.append(current_doc)

    # Apply masking only on train set and only if use_pretrained is True
    if use_pretrained and "train" in filename:
        documents = mask_10_percent(documents, mask_ratio=0.1)

    return documents
def extract_window_and_prefix_suffix_label_pairs(doc, split, window_size=2):
    if split == "test":
        tokens = doc
    else:
        tokens = [tok for tok, _ in doc]
        labels = [label for _, label in doc]

    padded_prefix_tokens = ["<PAD>"] * window_size + [token[:3] for token in tokens] + ["<PAD>"] * window_size
    padded_suffix_tokens = ["<PAD>"] * window_size + [token[-3:] for token in tokens] + ["<PAD>"] * window_size
    padded_tokens = ["<PAD>"] * window_size + tokens + ["<PAD>"] * window_size

    prefix_pairs = []
    suffix_pairs = []
    pairs = []

    if split == "test":
        for i in range(len(tokens)):
            prefix_window = padded_prefix_tokens[i:i + 2 * window_size + 1]
            suffix_window = padded_suffix_tokens[i:i + 2 * window_size + 1]
            window = padded_tokens[i:i + 2 * window_size + 1]

            pairs.append(window)  # label is for the center token
            prefix_pairs.append(prefix_window)
            suffix_pairs.append(suffix_window)
    else:
        for i in range(len(tokens)):
            window = padded_tokens[i:i + 2 * window_size + 1]
            prefix_window = padded_prefix_tokens[i:i + 2 * window_size + 1]
            suffix_window = padded_suffix_tokens[i:i + 2 * window_size + 1]
            pairs.append((window, labels[i]))  # label is for the center token
            prefix_pairs.append((prefix_window, labels[i]))
            suffix_pairs.append((suffix_window, labels[i]))
    return pairs, prefix_pairs, suffix_pairs

def extract_window_and_chars_label_pairs(doc, split, window_size=2):
    if split == "test":
        tokens = doc
    else:
        tokens = [tok for tok, _ in doc]
        labels = [label for _, label in doc]


    padded_tokens = ["<PAD>"] * window_size + tokens + ["<PAD>"] * window_size
    padded_chars = []
    for token in padded_tokens:
        padded_chars.append([["<PAD>"] * window_size + list(token) + ["<PAD>"] * window_size])

    chars_pairs = []
    pairs = []

    if split == "test":
        for i, token in enumerate(tokens):
            chars_window = padded_chars[i:i + 2 * window_size + 1]
            window = padded_tokens[i:i + 2 * window_size + 1]
            chars_pairs.append(chars_window)
            pairs.append(window)  # label is for the center token
            #chars_pairs.append(["<PAD>"] * window_size + list(token) + ["<PAD>"] * window_size)
    else:
        for i, token in enumerate(tokens):
            window = padded_tokens[i:i + 2 * window_size + 1]
            chars_window = padded_chars[i:i + 2 * window_size + 1]
            pairs.append((window, labels[i]))  # label is for the center token
            #chars_pairs.append((["<PAD>"] * window_size + list(token) + ["<PAD>"] * window_size, labels[i]))
            chars_pairs.append((chars_window, labels[i]))
    return pairs, chars_pairs


def extract_window_label_pairs(doc, split, window_size=2):
    if split == "test":
        tokens = doc
    else:
        tokens = [tok for tok, _ in doc]
        labels = [label for _, label in doc]
    
    padded_tokens = ["<PAD>"] * window_size + tokens + ["<PAD>"] * window_size

    pairs = []
    if split == "test":
        for i in range(len(tokens)):
            window = padded_tokens[i:i + 2 * window_size + 1]
            pairs.append(window)  # label is for the center token
    else:
        for i in range(len(tokens)):
            window = padded_tokens[i:i + 2 * window_size + 1]
            pairs.append((window, labels[i]))  # label is for the center token
    return pairs

def get_data_path(folder_name, task):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, '..', task, folder_name))
    return data_dir

def arrange_data(use_pretrained):
    documents = {}
    for task in ['ner', 'pos']:
        documents[task] = {
            'train': load_documents(get_data_path('train', task=task), use_pretrained),
            'dev': load_documents(get_data_path('dev', task=task), use_pretrained),
            'test': load_documents(get_data_path('test', task=task), use_pretrained)
        }
    all_window_data = {}
    for task in ['ner', 'pos']:
        all_window_data[task] = {}
        for split in ['train', 'dev', 'test']:
            all_pairs = []
            for doc in documents[task][split]:
                pairs = extract_window_label_pairs(doc, split, window_size=WINDOW_SIZE)
                all_pairs.extend(pairs)
            all_window_data[task][split] = all_pairs
    return all_window_data

def arrange_data_and_prefix_suffix(use_pretrained):
    documents = {}
    for task in ['ner', 'pos']:
        documents[task] = {
            'train': load_documents(get_data_path('train', task=task), use_pretrained),
            'dev': load_documents(get_data_path('dev', task=task), use_pretrained),
            'test': load_documents(get_data_path('test', task=task), use_pretrained)
        }
    all_window_data = {}
    for task in ['ner', 'pos']:
        all_window_data[task] = {}
        for split in ['train', 'dev', 'test']:
            all_pairs = []
            all_prefix_pairs = []
            all_suffix_pairs = []
            for doc in documents[task][split]:
                pairs, prefix_pairs, suffix_pairs = extract_window_and_prefix_suffix_label_pairs(doc, split, window_size=WINDOW_SIZE)
                all_pairs.extend(pairs)
                all_prefix_pairs.extend(prefix_pairs)
                all_suffix_pairs.extend(suffix_pairs)
            all_window_data[task][split] = (all_pairs, all_prefix_pairs, all_suffix_pairs)
    return all_window_data

def arrange_data_and_chars(use_pretrained):
    documents = {}
    for task in ['ner', 'pos']:
        documents[task] = {
            'train': load_documents(get_data_path('train', task=task), use_pretrained),
            'dev': load_documents(get_data_path('dev', task=task), use_pretrained),
            'test': load_documents(get_data_path('test', task=task), use_pretrained)
        }
    all_window_data = {}
    for task in ['ner', 'pos']:
        all_window_data[task] = {}
        for split in ['train', 'dev', 'test']:
            all_pairs = []
            all_chars_pairs = []
            for doc in documents[task][split]:
                pairs, chars_pairs = extract_window_and_chars_label_pairs(doc, split, window_size=WINDOW_SIZE)
                all_pairs.extend(pairs)
                all_chars_pairs.extend(chars_pairs)
            all_window_data[task][split] = (all_pairs, all_chars_pairs)
    return all_window_data


def build_label_vocab(pairs):
    label_vocab = {}
    for _, label in pairs:

    return label_vocab

def build_vocab(pairs):
    vocab = {'<PAD>': 0, '<UNK>': 1}

    for window, _ in pairs:
        for token in window:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

def build_vocab_chars(pairs):
    # vocab = {'<pad>': 0, '<unk>': 1}
    vocab = {'<PAD>': 0, '<UNK>': 1}

    for window, _ in pairs:
        for token in window:
            word=token[0]
            for char in word[2:-2]:
                # token = token.lower()
                if char not in vocab:
                    vocab[char] = len(vocab)
    return vocab


def lowercase_dataset(dataset_split):
    return [
        ([token.lower() for token in window], label)
        for window, label in dataset_split
    ]



def load_embed_vocab(file_name):
    """
    We added pad and unknown words to the embedding vocab
    """
    path = get_data_path(file_name, "embeddings")
    vocab_list = []
    with open(path, "r") as f:
        vocab_list = [line.strip().lower() for line in f if line.strip()]
    
    # Always add PAD and UNK if missing
    if "<PAD>" not in vocab_list:
        vocab_list = ["<PAD>"] + vocab_list
    if "<UNK>" not in vocab_list:
        vocab_list.insert(1, "<UNK>")  # insert after pad

    return {word: idx for idx, word in enumerate(vocab_list)}

def word2idx_embed_vocab(file_name):
    """
    Loads a vocabulary file where each line is a word.
    Returns a dictionary: word -> index
    """
    path = get_data_path(file_name, "embeddings")
    vocab_list = []
    with open(path, "r") as f:
        vocab_list = [line.strip().lower() for line in f if line.strip()]
    
    # Always add PAD and UNK
    if "<pad>" not in vocab_list:
        vocab_list = ["<pad>"] + vocab_list
    if "<unk>" not in vocab_list:
        vocab_list.insert(1, "<unk>")  # insert after pad

    return {word: idx for idx, word in enumerate(vocab_list)}

def load_pretrained_embeddings(file_name):
    """
    Loads pre-trained embeddings and prepends PAD and UNK.
    """
    path = get_data_path(file_name, "embeddings")
    vectors = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue  # skip empty lines
            vector = list(map(float, parts))
            vectors.append(vector)

    embedding_tensor = torch.tensor(vectors, dtype=torch.float)
    embedding_dim = embedding_tensor.shape[1]

    # Create <pad> and <unk> embeddings
    pad_vector = torch.zeros(embedding_dim)
    unk_vector = torch.randn(embedding_dim) * 0.6  # random small noise (or zeros if you prefer)

    # Add at the beginning
    full_tensor = torch.vstack([pad_vector, unk_vector, embedding_tensor])

    return full_tensor
