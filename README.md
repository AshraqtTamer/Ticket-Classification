# Multi-Language Ticket Classification System

This repository implements a Machine Learning pipeline to automatically categorize support tickets into different types (e.g., Request, Problem, Information Seeking). The project features advanced text preprocessing, language detection, and a comparative analysis between **Support Vector Machines (SVM)** and **XGBoost**.

## 📌 Project Overview

Efficiency in customer support often hinges on how quickly a ticket is routed to the correct department. This project automates that routing by analyzing the text body of multi-language tickets.

**Key stages include:**

1. **Language Filtering**: Using `langdetect` to identify and remove specific languages (e.g., German) to focus the model's scope.
2. **Text Refinement**: Comprehensive cleaning including digit removal, punctuation stripping, and NLTK-based Lemmatization.
3. **Class Balancing**: Utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the models are not biased toward more frequent ticket types.
4. **Vectorization**: Converting text into numerical data using **TF-IDF**.

## 🛠️ Tech Stack

* **Core**: Python, Pandas, NumPy
* **NLP**: `nltk` (WordNetLemmatizer, stopwords), `langdetect`
* **Machine Learning**: `scikit-learn` (SVM, TfidfVectorizer, LabelEncoder)
* **Boosting**: `XGBoost`
* **Imbalanced Data**: `imblearn` (SMOTE)
* **Visualization**: Seaborn, Matplotlib

## 📊 Model Performance & Comparison

The project compares two high-performance classifiers. Based on the evaluation metrics, **XGBoost** outperformed SVM in overall accuracy and handling the nuances between ticket categories.

| Metric | SVM (Linear Kernel) | XGBoost Classifier |
| --- | --- | --- |
| **Accuracy** | 85.15% | **88.13%** |
| **Class 0 (F1-Score)** | 0.98 | 0.97 |
| **Class 1 (F1-Score)** | 0.70 | **0.79** |
| **Class 2 (F1-Score)** | 0.74 | **0.79** |
| **Class 3 (F1-Score)** | 0.98 | 0.98 |

### Confusion Matrix Insights

While both models excel at identifying specific categories (Class 0 and 3), the Confusion Matrix reveals that XGBoost provides better precision and recall for the more challenging middle categories (Class 1 and 2).

## 🚀 Key Workflow Steps

### 1. Preprocessing & Cleaning

The `clean_text` function ensures that the model focuses on semantic meaning rather than noise:

* Lowercasing and digit removal.
* Stopword removal (e.g., "the", "is", "at").
* Lemmatization (reducing "requesting" or "requested" to the root word "request").

### 2. Handling Data Imbalance (SMOTE)

To prevent the model from simply predicting the most common class, SMOTE synthesizes new examples for minority classes in the feature space. This significantly improved the F1-scores for the less frequent ticket types.

### 3. Training

* **SVM**: Used a linear kernel for high-dimensional text efficiency.
* **XGBoost**: Utilized gradient boosting trees to capture non-linear relationships in the TF-IDF features.

## 🏗️ How to Run

1. **Install Dependencies**:
```bash
pip install pandas sklearn imbalanced-learn langdetect tqdm nltk xgboost

```


2. **Download NLTK Data**:
The script automatically downloads `punkt`, `stopwords`, and `wordnet`.
3. **Execute the Pipeline**:
Run the provided Python script or Notebook cells to process the data, train the models, and generate the evaluation heatmaps.
