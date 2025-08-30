# 📰 Fake News Detection App

A machine learning web app built with **Streamlit** that detects whether a given news article is **FAKE** or **REAL**.  
It uses NLP techniques such as **TF-IDF**, **text preprocessing**, and **sentiment analysis** to analyze input text and predict its authenticity.

## 🚀 Features
- Simple Streamlit interface for entering news text  
- Text preprocessing (tokenization, stopword removal, stemming/lemmatization, etc.)  
- Readability analysis using `textstat`  
- Sentiment analysis with NLTK’s VADER  
- Machine learning backend trained with `scikit-learn`  
- Visualization support with Plotly  

## 🛠️ Tech Stack
- **Frontend:** Streamlit  
- **Backend:** scikit-learn, pandas, numpy  
- **NLP:** NLTK, textstat, contractions, ftfy  

## 📦 Installation

Clone this repository:
```bash
git clone https://github.com/RohanKiani/fake-news-app.git
cd fake-news-app

pip install -r requirements.txt
streamlit run app/main.py
