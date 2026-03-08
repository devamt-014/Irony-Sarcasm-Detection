# Irony & Sarcasm Detection using Machine Learning and Deep Learning

This project explores different approaches for detecting irony and sarcasm in text using Natural Language Processing (NLP).

The system compares traditional machine learning models with deep learning and transformer-based models to evaluate their effectiveness in identifying figurative language.

---

## Project Overview

Sarcasm and irony detection is a challenging NLP task because the literal meaning of a sentence often differs from its intended meaning.

This project implements multiple models to classify tweets as:

* **Figurative (Irony/Sarcasm)**
* **Non-Figurative (Literal)**

The goal is to compare model performance and analyze which architecture works best for sarcasm detection.

---

## Models Implemented

The following models are implemented and evaluated:

### 1. Logistic Regression (Baseline)

* TF-IDF vectorization
* N-gram features (1–2 grams)
* Classical machine learning baseline

### 2. BiLSTM Neural Network

* Embedding layer
* Bidirectional LSTM
* Dropout regularization
* Binary classification output

### 3. DistilBERT Transformer

* Pretrained `distilbert-base-uncased`
* Fine-tuned for sarcasm detection
* Context-aware embeddings

---

## Dataset Processing

The dataset undergoes several preprocessing steps:

* Remove URLs
* Remove Twitter mentions
* Remove hashtags
* Remove duplicate tweets
* Remove extremely short tweets
* Convert labels into binary format

Classes are mapped as:

```
figurative / irony / sarcasm → 1
literal / non-figurative → 0
```

---

## Project Structure

```
sem_analysis
│
├── datasets
│   └── cleaned_dataset.csv
│
├── clean_dataset.py
├── baseline_lr.py
├── train_bilstm.py
├── train.py
├── distilbert_baseline.py
├── server_compare.py
│
├── all_model_metrics.csv
├── requirements.txt
└── README.md
```

---

## Training the Models

### 1. Clean the dataset

```
python clean_dataset.py
```

This generates the cleaned dataset used for training.

---

### 2. Train Logistic Regression

```
python baseline_lr.py
```

---

### 3. Train BiLSTM Model

```
python train_bilstm.py
```

---

### 4. Train DistilBERT Model

```
python train.py
```

---

## Model Comparison

All models log their results into:

```
all_model_metrics.csv
```

This file stores:

* Accuracy
* Precision
* Recall
* F1 score

for easy comparison across models.

---

## Running the API Server

You can start the comparison API using FastAPI:

```
uvicorn server_compare:app --host 0.0.0.0 --port 8001
```

Example request:

```
POST /compare
{
"text": "Great, my internet stopped working again."
}
```

Response:

```
{
  "baseline_sentiment": "Positive",
  "irony_label": "Ironic",
  "irony_prob": 0.71
}
```

---

## Technologies Used

* Python
* PyTorch
* Transformers (HuggingFace)
* Scikit-learn
* FastAPI
* Pandas
* NumPy

---

## Future Improvements

* Larger sarcasm datasets
* Context-aware sarcasm detection
* Multilingual irony detection
* Transformer ensemble models

---

## Author

Devam Trivedi
