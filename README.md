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
All taggers accept the following flags:

```
--task <ner|pos>
--use_pretrained <True|False>   # if supported by that part
```

### Part 1 – Window-Based Tagger
```
python tagger1.py --task <ner|pos>

```

### Part 2 – Most Similar Words
```
python top_k.py --word <WORD> --k 5
```

### Part 3 – Tagger with Pre-trained Embeddings
```
python tagger1.py --task <ner|pos> --use_pretrained True

```

### Part 4 – Tagger with Subword Units
```
python tagger3.py --task <ner|pos> --use_pretrained <True|False>

```

### Part 5 – CNN-Based Subword Model
```
python tagger4.py --task <ner|pos>
```
optional flag:

```
python tagger4.py --task <ner|pos> --use_pretrained False
```

### Part 6 – Character-Based Language Model
```
python langmodel.py --train corpus.txt --k <n> --sample_length <n>

```


---

## Notes
- Taggers use an MLP with tanh activation.
- Subword features include 3-letter prefixes and suffixes.
- CNN-based features follow Ma & Hovy (2016).
- NER accuracy is computed ignoring “O” tag matches as required.
- The sampling function generates character-level text sequences.
