# Emotion Recognition with TF-IDF and POS-Tag Features

## 📌 Project Overview
This project focuses on **emotion recognition** in text using the `dair-ai/emotion` dataset.  
We compare two models:
- **Baseline Model** → Uses only **TF-IDF features** with a Naive Bayes classifier.  
- **POS-Tag-Enhanced Model** → Combines **TF-IDF embeddings** with **Part-of-Speech (POS) tag features** extracted using the **Viterbi algorithm**.

The goal is to evaluate whether incorporating POS tag information improves classification performance.

---

## 📂 Dataset
We used the **`dair-ai/emotion` dataset** from Hugging Face, containing Twitter messages labeled with **six emotions**:
- Joy
- Sadness
- Anger
- Fear
- Love
- Surprise  

**Data Split:**
- Training: 80%  
- Validation: 10%  
- Test: 10%  

---

## ⚙️ Methodology
1. **Baseline Model**  
   - Extracted **TF-IDF features** from text.  
   - Trained a **Naive Bayes classifier**.  

2. **POS-Tag-Enhanced Model**  
   - Generated **POS tags** using a custom Viterbi-based tagger.  
   - Calculated frequency of each POS tag per sentence.  
   - Combined TF-IDF and POS-tag features using `numpy.hstack`.  
   - Trained a **Naive Bayes classifier** on combined features.  

---

## 📊 Results
**Baseline Model (Test Accuracy: 76%)**
- Higher **precision (0.81)** and **F1-score (0.73)**.  
- Performs better overall compared to POS-Tag model.  

**POS-Tag-Enhanced Model (Test Accuracy: 74%)**
- Slightly higher **recall (0.74 vs 0.76)** in some classes.  
- Struggles in certain emotions (e.g., Surprise).  
- Did not outperform the baseline overall.  

**Conclusion:**  
- Adding POS tags improved recall for some classes but did not significantly enhance overall performance.  
- More advanced models (e.g., neural networks) may leverage POS features better.  

---

## 🛠️ Installation & Usage
### Requirements
- Python 3.8+  
- Jupyter Notebook  
- Required libraries:  
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn nltk
  ```

### Run the Project
1. Open the Jupyter notebook:  
   ```bash
   jupyter notebook nlp_assign_1.ipynb
   ```
2. Run cells step by step to:
   - Load dataset
   - Train baseline and POS-tag models
   - Evaluate results with classification reports & confusion matrices  

---

## 📌 Files
- **`nlp_assign_1.ipynb`** → Implementation code  
- **`NLP_Assignment_1_24EE65R24.pdf`** → Report with methodology & results  
- **`README.md`** → Project documentation (this file)  

---

## 📢 Acknowledgments
- Dataset: [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)  
- POS Tagger: Custom implementation using **Viterbi algorithm**  
- Classifier: **Naive Bayes** from scikit-learn  
