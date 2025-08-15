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
