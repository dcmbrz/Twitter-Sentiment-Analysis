# Twitter Sentiment Analysis (Sentiment140)

Classify tweet sentiment using a clean, reproducible NLP pipeline: **data loading → text cleaning → TF–IDF vectorization (unigrams & bigrams) → model training & evaluation**. The project is organized in a single Jupyter notebook for clarity and fast iteration.

> **Why this repo?** A compact, end-to-end template for sentiment analysis you can adapt to other text datasets with minimal changes.

## Features

- **Dataset:** Sentiment140 (tweets + polarity labels)
- **Preprocessing:** text cleaning pipeline (e.g., lowercasing, punctuation/URL/user handle handling)
- **Vectorization:** `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`
- **Models:**
  - **Bernoulli Naive Bayes** (with **binary** features)
  - **Multinomial Naive Bayes** (directly on TF–IDF)
  - **Linear SVM (`LinearSVC`)** – strong baseline for sparse text
- **Evaluation:** train/test split, accuracy, **precision/recall/F1** classification report

## Project Structure
Project Structure

Twitter-Sentiment-Analysis/
├─ main.ipynb               # end-to-end pipeline (run this)
├─ requirements.txt         # Python dependencies
└─ Sentiment140 dataset.zip # dataset used by the notebook

If your dataset file is kept elsewhere, update the path in the notebook’s load step.

Quickstart

1) Set up a virtual environment

# Windows (PowerShell)
python -m venv .venv
. .venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

2) Install dependencies

pip install -r requirements.txt

If Jupyter is not installed:

pip install notebook

3) Run the notebook

jupyter notebook

Open main.ipynb and run cells top‑to‑bottom.

Pipeline Overview

Load Sentiment140 → keep polarity (labels) & text columns

Clean text (normalize case, strip noise, etc.) → clean_text

Split into train/test (e.g., 80/20) with a fixed random_state for reproducibility

Vectorize text with TF–IDF (unigrams + bigrams, ≤ 5k features)

Train a classifier (BernoulliNB / MultinomialNB / LinearSVC)

Evaluate with accuracy and classification_report (precision, recall, F1)

Key Notebook Cells (reference)

Train/Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42,
    stratify=df['polarity']  # preserves class balance
)

TF–IDF Vectorization

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,
    strip_accents='unicode'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

Models

BernoulliNB (binarized features):

from sklearn.naive_bayes import BernoulliNB

Xtr_bin = (X_train_tfidf > 0).astype(int)
Xte_bin = (X_test_tfidf  > 0).astype(int)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(Xtr_bin, y_train)

MultinomialNB (directly on TF–IDF):

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_tfidf, y_train)

Linear SVM (strong baseline):

from sklearn.svm import LinearSVC

svc = LinearSVC(C=1.0)
svc.fit(X_train_tfidf, y_train)

Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

pred = svc.predict(X_test_tfidf)  # or bnb_pred/mnb_pred
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
# Optional:
print(confusion_matrix(y_test, pred))

Labels: Binary vs. 3‑Class

Sentiment140 conventions often use 0 = negative, 4 = positive (some versions include 2 = neutral).

3‑class: keep 0, 2, 4 as‑is.

Binary: filter to {0,4} and map to {0,1}.

Example (binary):

mask_tr = y_train.isin([0,4])
mask_te = y_test.isin([0,4])

X_train_bi, y_train_bi = X_train[mask_tr], y_train[mask_tr].map({0:0, 4:1})
X_test_bi,  y_test_bi  = X_test[mask_te],  y_test[mask_te].map({0:0, 4:1})

# Refit vectorizer on the filtered training set
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train_bi)
X_test_tfidf  = vectorizer.transform(X_test_bi)

Tips & Gotchas

No data leakage: fit the vectorizer on train only; transform train & test with the same vectorizer.

BernoulliNB expects binary features: binarize TF–IDF (>0 → 1) or use binarize=0.0 in the estimator.

Linear models excel on sparse text: LinearSVC is a very strong baseline for TF–IDF.

Tune hyperparameters: try C ∈ {0.5, 1, 2} for SVM; adjust alpha for NB; increase max_features if RAM allows (e.g., 10k–50k).

Vectorizer hygiene: sublinear_tf=True, min_df=2, max_df=0.95 often help robustness.

Class imbalance: if present, consider class_weight='balanced' in linear models.

Reproducibility

Fixed split via random_state=42

Deterministic preprocessing steps in the notebook

Same vocabulary/IDF used for both train & test transforms

Results

Add your final metrics here after you run the notebook, for example:

LinearSVC (TF–IDF 1–2g, 5k feats)
Accuracy: 0.80
               precision    recall  f1-score   support
0                 0.80       0.78      0.79     ...
1                 0.79       0.81      0.80     ...

(Replace with your actual output.)

Troubleshooting

Out of memory: reduce max_features, use dtype=np.float32, or sample the data.

Weak scores: verify no leakage; try LinearSVC, tweak vectorizer; check cleaning (don’t remove negations like "not").

BernoulliNB underperforms: ensure features are binarized.

Roadmap

Grid/Randomized hyperparameter search

Stratified cross‑validation with confidence intervals

Emoji/hashtag/URL-aware cleaning; lemmatization

Class‑imbalance techniques; error analysis with confusion matrices

Transfer learning with transformer encoders (e.g., BERT) for improved accuracy
