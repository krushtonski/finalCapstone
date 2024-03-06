### finalCapstone - Sentiment Analysis of Amazon Product Reviews

This Python script performs sentiment analysis on product reviews on Amazon using NLP and gives the sentiment polarity (positive or negative) of review text and also shows the similarit between two different reviews. It uses SpaCy and Spacytextblob libraries. 

### Pre requisites

You need to have the following installed on your Terminal or equivalent:

### spacy for NLP
python -m spacy download en_core_web_md

### spacytextblob for sentiment analysis
# Install spacytextblob 
pip install spacytextblob 
python -m textblob.download_corpora  

### pandas for data visualisation and analysis 
Install pandas

Additionally, you need to download the spaCy English language models:

python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md

### How to use it:

1. Download the [Data Set](https://www.kaggle.com/code/weirditya/amazon-review-sentiment-analysis-using-python-ml) from Kaggle and rename one of the csv files to amazon_product_reviews.csv
2. Clone the repository:
git clone - (https://github.com/krushtonski/finalCapstone.git)
cd finalCapstone
3. Run the script:
python sentiment_analysis.py

### What you can see:
Sentiment Analysis: Understand the sentiment polarity of individual Amazon product reviews - negative or positive
Review Similarity Comparison: Compare the similarity between two selected reviews


### Images
