# Window-based-Tagging-and-ngram-Languge-Modeling

## Overview
This repository contains:
- Window-based POS/NER tagging
- Pre-trained embeddings
- Subword units (prefix/suffix)
- CNN-based character embeddings
- Character-based n-gram language modeling and sampling
---

## How to Run

### Part 1 – Window-Based Tagger
```
python tagger1.py --train train.txt --dev dev.txt --test test.txt --output output_file
```

### Part 2 – Most Similar Words
```
python top_k.py --word <WORD> --k 5
```

### Part 3 – Tagger with Pre-trained Embeddings
```
python tagger2.py --train train.txt --dev dev.txt --test test.txt                   --vectors vectors.txt --vocab words.txt
```

### Part 4 – Tagger with Subword Units
```
python tagger3.py --train train.txt --dev dev.txt --test test.txt --use_subwords
```

### Part 5 – CNN-Based Subword Model
```
python tagger4.py --train train.txt --dev dev.txt --test test.txt --use_cnn
```

### Part 6 – Character-Based Language Model
```
python langmodel.py --train corpus.txt --k 5 --sample_length 100
```

---

## Notes
- Taggers use an MLP with tanh activation.
- Subword features include 3-letter prefixes and suffixes.
- CNN-based features follow Ma & Hovy (2016).
- NER accuracy is computed ignoring “O” tag matches as required.
- The sampling function generates character-level text sequences.
