import numpy as np
from utils import get_data_path

# Load vectors and vocab
vocab_path = get_data_path("vocab.txt", "embeddings")
vectors_path = get_data_path("wordVectors.txt", "embeddings")
vectors = np.loadtxt(vectors_path)  # your actual vectors file

with open(vocab_path, "r") as f:
    vocab_list = [line.strip().lower() for line in f if line.strip()] #[line.strip() for line in f if line.strip()]

word2idx = {word: i for i, word in enumerate(vocab_list)}

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.sqrt(np.dot(u, u))
    norm_v = np.sqrt(np.dot(v, v))
    return dot_product / (norm_u * norm_v)

def most_similar(word, k=5):
    word = word.lower()
    if word not in word2idx:
        return f"The word '{word}' is not in vocab"

    word_idx = word2idx[word]
    word_vec = vectors[word_idx]

    similarities = []
    for second_word, second_word_idx in word2idx.items():
        if second_word == word:
            continue
        similarity = cosine_similarity(word_vec, vectors[second_word_idx])
        similarities.append((second_word, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

with open("part2.txt", "w") as f:
    for w in ["dog", "england", "john", "explode", "office"]:
        f.write(f"\nThe top-5 most similar words to '{w}':\n")
        results = most_similar(w, k=5)
        for similar_word, sim in results:
            f.write(f"  {similar_word}: the distance={sim:.3f}\n")


