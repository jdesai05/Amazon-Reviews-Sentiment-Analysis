# 📊 Amazon Review Sentiment Analysis

This project uses machine learning and natural language processing (NLP) techniques to analyze product reviews and predict whether a review is **positive** or **negative**. The final model is deployed through a user-friendly **Streamlit web application**.

---

## 🔍 Project Overview

- ✅ Dataset: Amazon Alexa Reviews (can generalize to other product reviews)
- 🧼 Preprocessing: Text cleaning, stopword removal, lemmatization
- 🧠 Model: Logistic Regression (trained on balanced data)
- 🧪 Vectorizer: TF-IDF (unigram + bigram support)
- 🗂️ Output: Trained model + TF-IDF vectorizer saved as `.pkl`
- 🌐 UI: Streamlit interface to input review text and view predictions

---

## 📁 File Structure

```bash
.
├── app.py                      # Streamlit app to run sentiment prediction
├── amazon_sentiment_model.ipynb  # Jupyter notebook with full pipeline
├── sentiment_model.pkl         # Saved ML model
├── tfidf_vectorizer.pkl        # Saved TF-IDF vectorizer
├── requirements.txt            # Python dependencies
└── README.md                   # You're here!
```

---

## 🧠 How It Works

1. **Load and Balance Dataset**
   - Original dataset is imbalanced (more positives than negatives)
   - Upsample the negative class to balance the distribution

2. **Clean the Text**
   - Lowercase conversion
   - Remove punctuation, numbers
   - Tokenize → Remove stopwords → Lemmatize using WordNet

3. **Vectorize using TF-IDF**
   - Max 5000 features
   - Use unigrams + bigrams
   - Outputs a sparse matrix for ML model

4. **Train Model**
   - Logistic Regression (for speed and interpretability)
   - Trained on the TF-IDF features

5. **Export**
   - Use `joblib` to save model and vectorizer as `.pkl` files

6. **Deploy with Streamlit**
   - User enters a review
   - Model predicts and displays sentiment and confidence

---

## 🚀 Getting Started

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis
```

### 📦 2. Install Dependencies

Make sure you have Python 3.8+

```bash
pip install -r requirements.txt
```

> If you're using NLTK, also run this:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### ▶️ 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Example Predictions

| Original Review                                           | Predicted Sentiment | Confidence |
|-----------------------------------------------------------|---------------------|------------|
| *"I love how easy it is to use. Very helpful!"*           | Positive 😀          | 97.2%      |
| *"Doesn’t work as expected. Completely useless."*         | Negative 😞          | 92.4%      |
| *"Packaging was okay but product failed after 2 weeks."*  | Negative 😞          | 85.7%      |

---

## 💡 Possible Improvements

- Add more diverse datasets (Amazon, Yelp, Twitter, etc.)
- Use models like SVM, XGBoost, or BERT for better performance
- Deploy on **Streamlit Cloud** or **Render**
- Add explanation/interpretability (LIME/SHAP)
- Bundle cleaning pipeline into a `.py` module

---

