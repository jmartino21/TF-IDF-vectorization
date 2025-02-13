# Text Clustering Using LSA and LDA

## Overview
This project applies **unsupervised learning techniques** for clustering text using **Latent Semantic Analysis (LSA)** and **Latent Dirichlet Allocation (LDA)** on article titles from Indiegogo.

## Features
- **Text Preprocessing**: Cleans and tokenizes article titles.
- **TF-IDF Vectorization**: Converts text into numerical format.
- **LSA (Latent Semantic Analysis)**: Extracts topics using singular value decomposition.
- **LDA (Latent Dirichlet Allocation)**: Clusters topics using probabilistic modeling.
- **Heatmap Visualization**: Displays topic-word relationships.

## Installation
### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

## Dataset
This project requires **five Indiegogo CSV files** containing article titles. Ensure they are placed in the same directory as the script. If missing, provide a sample dataset.

## Usage
### Running the Script
Execute the script using:
```bash
python text_clustering_lsa_lda.py
```

### Steps Performed
1. Loads and preprocesses article titles.
2. Converts text to TF-IDF vectors.
3. Applies **LSA** and **LDA** to extract topics.
4. Displays **top topic words** and **topic heatmaps**.

## Output
- **Top topic words for LSA and LDA**.
- **Heatmap visualizing topic-word distribution**.

## Findings
- **LSA and LDA provided different topic structures**, highlighting key themes in the dataset.
- **LDA's probabilistic model was more interpretable**, while **LSA captured hidden relationships between words**.

## License
This project is open-source and available for modification and use.
