# Sentiment analysis of Amazon product reviews

# Make sure you have installed pandas, spacy and spacytextbob in the Terminal
"""Install pandas"""
""" python -m spacy download en_core_web_md"""
""" # Install spacytextblob""" 
""" pip install spacytextblob"""  
""" python -m textblob.download_corpora""" 

# Import the External libraries
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the small spacy model for NLP and add SpacyTextBlob component to perform sentiment analysis
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

# Define "cleandata" function to tokenize, lemmatize, and remove stopwords and punctuation
def cleandata(text):
    doc = nlp(text)
    return ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])

# Define "sentiment analysis" to analyse sentiment of Amazon reviews using SpacyTextBlob
# Extract polarity and sentiment scores
# Label negative sentiment as less than 0, positive as more than 0
# Create a dictionairy with sentiment analysis results 
def sentiment_anal(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity      
    sentiment = doc._.blob.sentiment   
    label = "positive" if polarity > 0 else ("negative" if polarity < 0 else "neutral")     
    return {"polarity": polarity, "sentiment": sentiment, "label": label}     


# Read the amazon_product_reviews file, select the reviews.text column and remove the empty cells
df = pd.read_csv('amazon_product_reviews.csv', sep=',')
df = df.dropna(subset=['reviews.text']) 
df = df[['reviews.text']]

# Select a random sample of 2 reviews to test the model
two_reviews = df.sample(2, random_state=10)

# Apply cleaning of data to the two sample reviews "two_reviews" for the review.text
two_reviews['cleaned_reviews'] = two_reviews['reviews.text'].apply(cleandata)

# Analyzing sentiment for the two sample review
sentiment_analysis = []
for review in two_reviews['cleaned_reviews']:
    sentiment_analysis.append(sentiment_anal(review))

# Adding sentiment analysis results as separate columns
two_reviews['polarity'] = [analysis['polarity'] for analysis in sentiment_analysis]
two_reviews['sentiment'] = [analysis['sentiment'] for analysis in sentiment_analysis]
two_reviews['label'] = [analysis['label'] for analysis in sentiment_analysis]

# Printing the resulting DataFrame with the desired columns
print("\n----------Sentiment Analysis Results----------")
print(two_reviews[['reviews.text', 'polarity', 'sentiment', 'label']])

# Loading the spaCy model with word vectors
nlp = spacy.load("en_core_web_md")

# Defining the two reviews to compare
first_review_1 = df['reviews.text'][55]
second_review_2 = df['reviews.text'][117]

# Preprocessing the reviews using spaCy
review1_doc = nlp(first_review_1)
review2_doc = nlp(second_review_2)

# Calculating the similarity score between the two reviews
similarity_score = review1_doc.similarity(review2_doc)

# Printing the similarity score
print("\n----------Similarity Score----------")
print("Similarity Score:", similarity_score)

#Printing the number of reviews in the amazon_product_reviews.csv file (number of rows - 1)
print("The number of reviews in this dataset:")
print(df['reviews.text'].count())