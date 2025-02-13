import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

# Load and preprocess data
def load_data(file_list):
    title_list = []
    for file in file_list:
        df = pd.read_csv(file)
        titles = df['title'].dropna().tolist()
        title_list.extend(titles)
    return title_list

# Text cleaning function
def clean_text(text_list):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    cleaned_titles = []
    for title in text_list:
        title = title.lower()
        title = re.sub(r'[^\w\s]', '', title)
        title = ' '.join([word for word in title.split() if word not in stop_words])
        cleaned_titles.append(title)
    return cleaned_titles

# TF-IDF Vectorization
def vectorize_text(cleaned_titles):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(cleaned_titles), vectorizer

# LSA Topic Modeling
def apply_lsa(tfidf_matrix, num_topics=5):
    lsa = TruncatedSVD(n_components=num_topics, random_state=42)
    return lsa.fit_transform(tfidf_matrix), lsa

# LDA Topic Modeling
def apply_lda(cleaned_titles, num_topics=5):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(cleaned_titles)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    return lda.fit_transform(dtm), lda, vectorizer

# Display top words per topic
def print_topics(model, vectorizer, num_words=10):
    terms = vectorizer.get_feature_names_out()
    for index, topic in enumerate(model.components_):
        print(f"Topic {index + 1}:", [terms[i] for i in topic.argsort()[:-num_words - 1:-1]])

# Heatmap visualization
def plot_heatmap(topic_matrix, title):
    plt.figure(figsize=(10, 5))
    sns.heatmap(topic_matrix, cmap='coolwarm', annot=False)
    plt.title(title)
    plt.show()

# Main script execution
if __name__ == "__main__":
    file_list = ['Indiegogo1.csv', 'Indiegogo2.csv', 'Indiegogo3.csv', 'Indiegogo4.csv', 'Indiegogo5.csv']
    titles = load_data(file_list)
    cleaned_titles = clean_text(titles)
    
    tfidf_matrix, vectorizer = vectorize_text(cleaned_titles)
    lsa_topics, lsa_model = apply_lsa(tfidf_matrix)
    lda_topics, lda_model, lda_vectorizer = apply_lda(cleaned_titles)
    
    print("Top LSA Topics:")
    print_topics(lsa_model, vectorizer)
    print("\nTop LDA Topics:")
    print_topics(lda_model, lda_vectorizer)
    
    plot_heatmap(lsa_topics, "LSA Topic Heatmap")
    plot_heatmap(lda_topics, "LDA Topic Heatmap")
