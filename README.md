# Sentiment Analysis: A Comparative Study of Machine Learning Models

## Table of Contents
- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Datasets Used](#datasets-used)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Implementation Steps](#implementation-steps)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)
- [Requirements](#requirements)

## Introduction
Sentiment analysis is a crucial Natural Language Processing (NLP) technique used to extract opinions and emotions from textual data. This project evaluates and compares various machine learning models for sentiment analysis across different domains, including movie reviews, product reviews, and social media tweets.

## Project Objectives
- Analyze and compare sentiment analysis models across different textual domains.
- Evaluate models using performance metrics such as accuracy, precision, recall, and F1-score.
- Identify the most effective models for different types of text data.
- Explore challenges in sentiment classification and provide recommendations for improvements.

## Datasets Used
Datasets were obtained from **Kaggle** and included:
1. **Movie Reviews** - Structured textual data with clear sentiment classification.
2. **Product Reviews** - User-generated reviews on various products.
3. **Tweets** - Social media posts containing informal and noisy text.

## Methodology
The methodology follows a systematic process:
1. **Data Collection** - Collecting labeled datasets from Kaggle.
2. **Preprocessing** - Cleaning and transforming text data using tokenization, stop word removal, lemmatization, and vectorization.
3. **Model Selection** - Training various machine learning models for sentiment analysis.
4. **Evaluation** - Comparing models using accuracy, precision, recall, and F1-score.
5. **Analysis** - Identifying the best-performing model for each domain.

## Models Implemented
- **Support Vector Machine (SVM)**
- **Naive Bayes (MultinomialNB)**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Classifier (GBC)**

## Implementation Steps
### Step 1: Data Collection
- Download datasets from Kaggle.
- Format data for training and testing.

### Step 2: Data Preprocessing
- Tokenization
- Stop word removal
- Lemmatization
- Vectorization using **TF-IDF** or **CountVectorizer**
- Data balancing and splitting (80% train, 20% test)

### Step 3: Model Selection and Training
- Train selected models on the processed dataset.
- Tune hyperparameters to optimize performance.

### Step 4: Model Evaluation
- Evaluate using accuracy, precision, recall, and F1-score.
- Compare performance across different textual domains.

## Results and Discussion
- **Movie Reviews:** Support Vector Machine (SVM) performed best with **94% accuracy**.
- **Product Reviews:** Multinomial Naive Bayes and SVM showed the highest accuracy of **90-93%**.
- **Tweets Sentiment:** Random Forest performed the best with **89% accuracy**, handling informal text better than other models.

## Conclusion
- **SVM** is the best choice for structured datasets like Movie and Product Reviews.
- **Random Forest** is more effective for informal and noisy text like Tweets.
- **Naive Bayes** is a strong contender in structured datasets but lags in social media sentiment analysis.
- **Deep learning models (LSTM, CNN, BERT)** can be explored for future improvements.

## Future Scope
- Implement advanced deep learning models such as **BERT, CNN, and LSTM**.
- Expand dataset size and domain diversity for better generalization.
- Fine-tune models using **transfer learning**.
- Develop real-time sentiment analysis applications (e.g., chatbot integrations, social media monitoring).

## Requirements
To run this project, ensure you have the following installed:
- Python 3.x
- Jupyter Notebook or any Python IDE
- Required Libraries:
  ```bash
  pip install numpy pandas sklearn nltk matplotlib seaborn tensorflow keras
  ```
