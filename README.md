# Fake News Detection

A machine learning project to classify news articles as **REAL** or **FAKE** using NLP and multiple models (Logistic Regression, Linear SVM, Passive-Aggressive, Naive Bayes). The notebook follows a simple, readable structure similar to YBI Foundation samples—upgraded with EDA, visualization, multiple algorithms, and clean evaluation.

---

## Problem Statement
Identify whether a news article is **fake** or **real** based on its textual content. This helps platforms, researchers, and readers flag misinformation.

---

## Dataset
We use a widely used Fake News dataset with columns like:
- `id` (optional)
- `title`
- `text`
- `subject` (optional)
- `date` (optional)
- `label` — typically `FAKE`/`REAL` or `0`/`1`


---

## Project Pipeline
1. **Import Libraries**  
2. **Load Data** (GitHub/Kaggle)  
3. **Quick EDA** (shape, nulls, class balance, word counts)  
4. **Preprocessing** (clean text: lowercasing, punctuation removal, stopwords, optional lemmatization)  
5. **Train/Test Split**  
6. **Vectorization** (TF-IDF with n-grams)  
7. **Modeling**  
   - Logistic Regression  
   - Linear SVM  
   - Passive-Aggressive Classifier  
   - Multinomial Naive Bayes  
8. **Evaluation** (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC where applicable)  
9. **Error Analysis** (sample misclassifications)  
10. **Save Best Model**  
11. **(Optional) Simple Inference Helper**

---

## Technology Used
- Python, Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- (Optional) NLTK for stopwords  
- Google Colab, GitHub

---

## Results (example)
- Linear SVM / Passive-Aggressive often perform best on TF-IDF text.
- Expect **90–98%** accuracy depending on dataset and preprocessing.

> Actual results will be printed by the notebook after training.



