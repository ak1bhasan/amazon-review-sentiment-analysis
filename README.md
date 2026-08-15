# Amazon Review Sentiment Analysis using RNN, LSTM, and GRU

## Overview

This project is a comparative study of three recurrent neural network architectures—**Simple RNN**, **LSTM**, and **GRU**—for binary sentiment classification of Amazon customer reviews.

Sentiment analysis helps automatically determine whether customer feedback is negative or positive, which is useful for understanding product reception at scale. The main goal of this experiment is to evaluate which recurrent architecture performs best under a shared preprocessing pipeline, embedding setup, training configuration, and evaluation protocol.

Under the final experimental configuration (padding masking + EarlyStopping), **GRU** achieved the highest test accuracy (**90.18%**), followed by LSTM (**88.86%**) and Simple RNN (**81.56%**).

---

## Objectives

Based on the notebook’s stated goals and experimental design, this project aims to:

1. Build a binary sentiment classifier for Amazon customer reviews (`0` = Negative, `1` = Positive).
2. Train and evaluate three recurrent architectures under comparable conditions:
   - Simple RNN
   - LSTM
   - GRU
3. Investigate the impact of padding masking (`mask_zero=True`) and EarlyStopping on model performance.
4. Compare the models using accuracy, precision, recall, F1-score, confusion matrices, and parameter counts.
5. Test the trained models on manually written custom reviews.

---

## Dataset

| Item | Details |
|---|---|
| Dataset name | Amazon Reviews |
| Source | [Kaggle — `bittlingmayer/amazonreviews`](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) |
| Original files | `train.ft.txt.bz2`, `test.ft.txt.bz2` (FastText-style labeled text) |
| Full training set size | 3,600,000 reviews (1,800,000 positive / 1,800,000 negative) |
| Sample used in this project | **50,000** reviews (`df.sample(50000, random_state=42)`) |
| Label mapping | `__label__1` → `0` (Negative), `__label__2` → `1` (Positive) |
| Sample class balance | 25,039 positive / 24,961 negative |

**Important:** The notebook downloads both train and test archives, but modeling uses a random sample from `train.ft.txt.bz2` only. That sample is then split into train/validation/test sets. The official Kaggle test file is not used for evaluation.

### Dataset characteristics (50,000-sample subset)

- No missing values
- No duplicate rows
- Review length (word count): mean ≈ 78.55, min = 13, max = 210

---

## Technologies and Libraries

Libraries and tools actually used in the notebook:

- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **scikit-learn** (`train_test_split`, classification metrics, confusion matrix)
- **TensorFlow / Keras** (`TextVectorization`, `Embedding`, `SimpleRNN`, `LSTM`, `GRU`, `Dropout`, `Dense`, `EarlyStopping`)
- **WordCloud** (EDA visualizations)
- **bz2** (reading compressed FastText files)
- **re** (text cleaning)
- **Kaggle API / CLI** (dataset download)

---

## Methodology

1. Download and extract the Amazon Reviews dataset from Kaggle.
2. Parse FastText labels and review text from `train.ft.txt.bz2`.
3. Map labels to binary sentiment (`0` / `1`) and sample 50,000 reviews.
4. Perform exploratory data analysis (class balance, review length, word clouds).
5. Clean and normalize review text.
6. Split data into train / validation / test sets (stratified).
7. Fit a `TextVectorization` vocabulary on training data only; convert texts to padded integer sequences.
8. Train initial unmasked Simple RNN and LSTM models.
9. Retrain with padding masking (`mask_zero=True`) and EarlyStopping for fair comparison.
10. Train the final masked GRU model under the same protocol.
11. Evaluate all final models on the held-out test set.
12. Compare metrics, confusion matrices, and parameter counts.
13. Run custom review predictions with the trained models.

---

## Text Preprocessing

The notebook applies the following preprocessing steps:

1. **Lowercasing** — `df['text'].str.lower()`
2. **HTML tag removal** — regex `<.*?>`
3. **Character filtering** — keep only letters and spaces (`[^a-zA-Z ]` replaced with space)
4. **Whitespace normalization** — collapse repeated whitespace and strip ends
5. **Empty-row removal** — drop rows with empty text after cleaning (none were found in the sample)

Tokenization / sequence preparation (after cleaning):

- Keras `TextVectorization` with `max_tokens = 20,000`
- Vocabulary fitted on **training data only** (`vectorizer.adapt(X_train)`)
- Output mode: integer token IDs
- Maximum sequence length: **200** (`output_sequence_length = 200`)
- Shorter sequences are zero-padded to length 200

---

## Data Split

Stratified splits with `random_state = 42`:

| Split | Size | Proportion of sample |
|---|---:|---:|
| Training | 40,000 | 80% |
| Validation | 5,000 | 10% |
| Test | 5,000 | 10% |

Procedure:

1. `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`
2. Split the remaining 20% evenly into validation and test (`test_size=0.50`, stratified, `random_state=42`)

Global reproducibility seed used in the notebook: `SEED = 42` (Python `random`, NumPy, and TensorFlow).

---

## Text Vectorization and Embedding

| Setting | Value |
|---|---|
| Vocabulary size | 20,000 |
| Sequence length | 200 |
| Embedding dimension | 128 |
| Vocabulary fitted on | Training set only |
| Final padding masking | `mask_zero=True` in the Embedding layer |

In the final models, padding tokens are masked so recurrent layers do not treat padding as meaningful content.

---

## Models

All final models share the same high-level structure:

`Input (200)` → `Embedding (20000 → 128)` → `Recurrent layer (64 units)` → `Dropout (0.3)` → `Dense (1, sigmoid)`

### Simple RNN

- Architecture: `Embedding` → `SimpleRNN(64)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- Final version uses `mask_zero=True`
- Total parameters: **2,572,417**

### LSTM

- Architecture: `Embedding` → `LSTM(64)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- Final version uses `mask_zero=True`
- Total parameters: **2,609,473**

### GRU

- Architecture: `Embedding` → `GRU(64)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
- Uses `mask_zero=True`
- Total parameters: **2,597,313**

### Conceptual comparison (project context)

- **Simple RNN** is the simplest recurrent baseline and can struggle with longer dependencies.
- **LSTM** adds gating mechanisms that help retain longer-range information; in this study it outperformed the Simple RNN.
- **GRU** is also gated, but structurally simpler than LSTM; in this study it achieved the best overall test performance.

---

## Training Configuration

Final training settings used for the comparable masked models:

| Setting | Value |
|---|---|
| Optimizer | Adam (default Keras learning rate; no custom LR specified) |
| Loss | `binary_crossentropy` |
| Metric | Accuracy |
| Batch size | 64 |
| Maximum epochs | 10 |
| EarlyStopping monitor | `val_loss` |
| Patience | 2 |
| `restore_best_weights` | `True` |
| Random seed | 42 |

Decision threshold for positive sentiment at inference: probability ≥ 0.5.

---

## Experimental Progression

This notebook is an experimental progression, not a single final run.

### Initial / Unmasked Experiments

Initial models used Embedding **without** `mask_zero=True` and trained for 5 epochs (batch size 64):

| Model | Test Accuracy | Observation |
|---|---:|---|
| Simple RNN (unmasked) | **65.58%** | Weak baseline; precision/recall/F1 ≈ 0.65 |
| LSTM (unmasked) | **50.14%** | Near-chance; predicted almost all samples as Positive (negative-class recall ≈ 0.00) |

Training/validation accuracy and loss curves are plotted for these initial unmasked RNN and LSTM runs.

### Intermediate Masked LSTM (no EarlyStopping)

An LSTM with `mask_zero=True` was then trained for 5 epochs without EarlyStopping. Validation accuracy improved substantially (best observed validation accuracy ≈ **89.40%** at epoch 3), confirming that padding masking was critical.

### Final / Masked Experiments (official comparison)

For fair comparison, the notebook retrained:

- Simple RNN with masking + EarlyStopping
- LSTM with masking + EarlyStopping
- GRU with masking + EarlyStopping

These are the **official final results** used for model ranking.

Padding masking substantially improved performance by preventing padded zeros from distorting recurrent computation.

---

## Results

### Final verified model comparison

Metrics below are taken from the notebook’s final comparison table (positive-class precision/recall/F1 via scikit-learn defaults, reported as percentages):

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|---|---:|---:|---:|---:|---:|
| Simple RNN | 81.56% | 83.38% | 78.91% | 81.08% | 2,572,417 |
| LSTM | 88.86% | 88.92% | 88.82% | 88.87% | 2,609,473 |
| GRU | **90.18%** | **91.68%** | **88.42%** | **90.02%** | 2,597,313 |

**Best model:** GRU (highest test accuracy and F1-score).

Final ranking:

1. GRU — 90.18%
2. LSTM — 88.86%
3. Simple RNN — 81.56%

### Per-class metrics (final models)

**Simple RNN**

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.80 | 0.83 |
| Recall | 0.84 | 0.79 |
| F1-score | 0.82 | 0.81 |

**LSTM**

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.89 | 0.89 |
| Recall | 0.89 | 0.89 |
| F1-score | 0.89 | 0.89 |

**GRU**

| Metric | Negative | Positive |
|---|---:|---:|
| Precision | 0.89 | 0.92 |
| Recall | 0.92 | 0.88 |
| F1-score | 0.90 | 0.90 |

---

## Initial Experimental Results

These belong to the earlier unmasked stage and should **not** be mixed with the final comparison:

| Model | Test Accuracy |
|---|---:|
| Initial Simple RNN (unmasked) | 65.58% |
| Initial LSTM (unmasked) | 50.14% |

---

## Confusion Matrix Analysis

Class ordering used in the notebook: **Negative (0)**, **Positive (1)**.

Matrices are in the form:

```text
[[TN, FP],
 [FN, TP]]
```

### Final Simple RNN

```text
[[2102, 394],
 [528, 1976]]
```

- Correct negatives: 2,102
- Correct positives: 1,976
- False positives: 394
- False negatives: 528

### Final LSTM

```text
[[2219, 277],
 [280, 2224]]
```

- Correct negatives: 2,219
- Correct positives: 2,224
- False positives: 277
- False negatives: 280

### Final GRU

```text
[[2295, 201],
 [290, 2214]]
```

- Correct negatives: 2,295
- Correct positives: 2,214
- False positives: 201
- False negatives: 290

GRU produced the strongest overall classification behavior among the three final models, with fewer total misclassifications than Simple RNN and a strong balance across both classes.

---

## Training and Validation Performance

Evidence available in the notebook:

### Curves plotted

- Initial / unmasked Simple RNN (accuracy and loss)
- Initial / unmasked LSTM (accuracy and loss)
- Final / masked GRU (accuracy and loss)

### Curves not plotted

Final masked Simple RNN and final masked LSTM have **training logs**, but the notebook does **not** include plotted training/validation curves for those two final runs.

### Logged final-run behavior (masked + EarlyStopping)

**Final LSTM**

| Epoch | Train Acc | Val Acc | Val Loss |
|---:|---:|---:|---:|
| 1 | 0.8404 | 0.8824 | 0.2857 |
| 2 | 0.9218 | 0.9000 | 0.2704 |
| 3 | 0.9397 | 0.8870 | 0.3234 |
| 4 | 0.9558 | 0.8856 | 0.3381 |

Stopped after epoch 4 (patience = 2); best weights restored from the lowest validation loss (epoch 2). Test accuracy: **88.86%**.

**Final Simple RNN**

| Epoch | Train Acc | Val Acc | Val Loss |
|---:|---:|---:|---:|
| 1 | 0.7029 | 0.7224 | 0.5489 |
| 2 | 0.8177 | 0.8140 | 0.4337 |
| 3 | 0.8476 | 0.8238 | 0.4330 |
| 4 | 0.9273 | 0.8370 | 0.4569 |
| 5 | 0.9683 | 0.8286 | 0.5631 |

Stopped after epoch 5; best weights restored from epoch 3 (lowest `val_loss`). Test accuracy: **81.56%**.

**Final GRU**

| Epoch | Train Acc | Val Acc | Val Loss |
|---:|---:|---:|---:|
| 1 | 0.8364 | 0.8966 | 0.2671 |
| 2 | 0.9277 | 0.9050 | 0.2542 |
| 3 | 0.9579 | 0.8976 | 0.3071 |
| 4 | 0.9750 | 0.8978 | 0.3741 |

Stopped after epoch 4; best weights restored from epoch 2 (lowest `val_loss`, highest validation accuracy ≈ **90.50%**). Test accuracy: **90.18%**.

After epoch 2, GRU training accuracy continued to rise while validation loss worsened, and EarlyStopping halted further training.

---

## Custom Predictions

The notebook tests **5** manually written reviews with the final masked models (`model_rnn_masked`, `model_lstm`, `model_gru`). No ground-truth labels are assigned to these custom inputs in the notebook.

| Review | RNN | LSTM | GRU |
|---|---|---|---|
| "This product is amazing. I really loved it!" | Positive (0.986) | Positive (0.958) | Positive (0.996) |
| "Terrible product. I regret buying it." | Negative (0.080) | Negative (0.079) | Negative (0.036) |
| "The product is okay, but nothing special." | Negative (0.093) | Negative (0.135) | Negative (0.097) |
| "Excellent quality and fast delivery." | Positive (0.984) | Positive (0.980) | Positive (0.995) |
| "Waste of money. Very disappointing." | Negative (0.002) | Negative (0.000) | Negative (0.000) |

Values in parentheses are predicted positive-class probabilities.

---

## Key Findings

1. **GRU performed best** on the final test set (90.18% accuracy, 90.02% F1-score).
2. **LSTM outperformed Simple RNN** in the final masked setting (88.86% vs 81.56%).
3. **Padding masking was critical**: unmasked LSTM collapsed to ~50% accuracy and nearly always predicted Positive; enabling `mask_zero=True` recovered strong performance.
4. **EarlyStopping** helped select better weights by monitoring `val_loss` and restoring the best checkpoint.
5. Gated recurrent models (LSTM/GRU) were more effective than the basic Simple RNN for this Amazon review sentiment task under the shared experimental setup.

---

## Limitations

As stated in the notebook:

1. Vocabulary size and maximum sequence length were fixed; other values may change results.
2. Only one main architecture configuration was evaluated for each recurrent model.
3. Embeddings were randomly initialized (no pretrained Word2Vec/GloVe/etc.).
4. Hyperparameter tuning was limited (learning rate, units, dropout, batch size, sequence length, etc.).
5. Evaluation used the project’s sampled/held-out test split, so results may not generalize to all review domains or the full Amazon Reviews corpus.
6. The study used a **50,000-review sample** rather than the full 3.6M training set.

---

## Future Work

Suggested extensions from the notebook (not implemented in this project):

1. Broader hyperparameter tuning for RNN, LSTM, and GRU.
2. Experiments with different embedding dimensions and sequence lengths.
3. Pretrained embeddings (e.g., Word2Vec or GloVe).
4. Bidirectional RNN / LSTM / GRU architectures.
5. Attention mechanisms.
6. Comparison against transformer-based models.
7. Evaluation on additional sentiment datasets to assess generalization.
8. Scaling experiments to larger portions of the Amazon Reviews corpus.

---

## Project Structure

```text
Amazon Review Sentiment Analysis/
│
├── Amazon_Review_Sentiment_Analysis_using_RNN,_LSTM,_and_GRU.ipynb
└── README.md
```

The notebook is self-contained: it downloads the dataset from Kaggle, performs preprocessing, trains the models, and reports evaluation results.

---

## How to Run

1. Open `Amazon_Review_Sentiment_Analysis_using_RNN,_LSTM,_and_GRU.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Ensure a Kaggle API credential file is available (the notebook expects Kaggle access to download `bittlingmayer/amazonreviews`).
3. Run all cells sequentially.
4. Review training logs, metric tables, confusion matrices, and custom prediction outputs.

**Note:** Training the recurrent models (especially LSTM/GRU) is computationally intensive and may take a substantial amount of time on CPU.

---

## Final Summary

| Item | Result |
|---|---|
| Task | Binary Amazon review sentiment classification |
| Sample size | 50,000 reviews |
| Models compared | Simple RNN, LSTM, GRU |
| Best model | **GRU** |
| Best test accuracy | **90.18%** |
| Final ranking | GRU > LSTM > Simple RNN |
