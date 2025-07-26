# 📰 Fake News Detection Using Machine Learning & Deep Learning

![Notebook](https://img.shields.io/badge/Tool-Jupyter_Notebook-orange.svg)

A machine learning and deep learning project aimed at detecting fake news articles using various classification models such as SVC, XGBoost, and RNN (LSTM). The project addresses the complex challenges of misinformation using both traditional and neural network-based NLP techniques.


## 📂 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Features](#-features)
- [Technologies](#-technologies)
- [Results](#-results)
- [Contact](#-contact)

---

## 🧠 Problem Statement

The widespread dissemination of fake news and propaganda presents serious societal risks, including:

- Erosion of public trust
- Political polarization and manipulation of elections
- Spread of harmful misinformation during crises (e.g., pandemics, conflicts)

From an NLP perspective, fake news detection is challenging due to:

- Linguistic mimicry of real journalism
- Lack of reliable, up-to-date labeled datasets across languages and regions
- Evolving adversarial strategies used by malicious actors
- Complications from sarcasm, satire, cultural context, and implicit bias
- Risk of model bias and over-censorship in automated detection systems

These issues call for context-aware, robust, and ethically responsible solutions.

---

## 📁 Dataset

📎 **Download Link:**  
[Google Drive - Dataset Folder](https://drive.google.com/drive/folders/1mrX3vPKhEzxG96OCPpCeh9F8m_QKCM4z?usp=sharing)

### ✅ Real News  
**File:** `MisinfoSuperset_TRUE.csv`  
**Sources:** Reputable media outlets like *Reuters, The New York Times, The Washington Post*, etc.

### ❌ Fake/Misinformation/Propaganda  
**File:** `MisinfoSuperset_FAKE.csv`  
**Sources:**  
- Right-wing extremist sources: *Breitbart, Redflag Newsdesk, Truth Broadcast Network*  
- From: Ahmed, H., Traore, I., & Saad, S. (2017), *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques*, Springer LNCS 10618

---

## ✨ Features

### 1. 🧹 Data Preprocessing
- Clean and normalize text (lowercasing, removing punctuation, stopwords, etc.)
- Combine `TRUE` and `FAKE` datasets with labels
- Tokenization, stemming/lemmatization

### 2. 📊 Exploratory Data Analysis (EDA)
- Visualize word distributions, class balance
- Generate word counts for fake vs. real news
- Analyze article lengths and token frequencies

### 3. Classical ML Models (SVC, XGBoost)
- Build models using TF-IDF and n-gram features
- Train **Support Vector Classifier (SVC)** and **XGBoost Classifier**
- Evaluate performance using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix

### 4. Deep Learning Model - LSTM (RNN)

- Built a Recurrent Neural Network (RNN) using Long Short-Term Memory (LSTM) layers to capture sequential and contextual dependencies in the text data.

- Utilizing pre-trained word embeddings from spaCy to initialize the embedding layer, improving word preresentation and reducing training time.

```markdown
### Model Flow
[Input Text]
     ↓
[Embedding Layer]
(spaCy vectors)
     ↓
[ LSTM Layer ]
     ↓
[Fully Connected Layer]
     ↓
[Sigmoid Output]
```

### 5. Model Training
- Train models on training set with validation
- Monitor loss, accuracy, and performance metrics

### 6. 📈 Performance Evaluation
- Evaluate on unseen test data
- Compare performance of classical vs. deep learning models
- Visualize confusion matrix, precision-recall tradeoffs

---

## 🛠️ Technologies

- Python 3.10, Jupyter Notebook
- `pandas`, `numpy` for data processing
- `matplotlib`, `seaborn` for visualization
- Machine Learning : `scikit-learn`, `XGBoost`
- Natural Language Processing : `nltk`, `re`
- Deep Learning : 
  - Tokenizer, pad_sequences :  `tensorflow.keras.preprocessing`
  - Building and training LSTM model (RNN) : `pytorch`
  - Pre-trained Embeddings (en_core_web_sm) : `spacy` 
  
---

## ✅ Results

- **XGBoost** and **SVC** offer fast and explainable results with good performance on structured textual data.
- **LSTM** captures sequential and contextual patterns but requires more training time.
- Overall performance (sample):
  
```markdown
| Model     | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| SVC       | 0.94     | 0.94      | 0.94   | 0.94     |
| XGBoost   | 0.95     | 0.95      | 0.95   | 0.95     |
| LSTM      | 0.96     | 0.96      | 0.96   | 0.96     |

```

---

## 📬 Contact
Nguyễn Văn Trung Chính – trungchinh19082004@gmail.com
GitHub: TrungChinh-Byte
