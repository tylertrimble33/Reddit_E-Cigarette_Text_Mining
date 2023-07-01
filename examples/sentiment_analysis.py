import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# download the vader lexicon (need to do once)
nltk.download('vader_lexicon')
# download punkt for the tokenizer (need to do once)
nltk.download('punkt')

# Create an instance of the SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Sample text for sentiment analysis
text_pos = "I LOVE my vape pen. It is so cool"
text_neg = "Vaping is HORRIBLE for you!"
text_neutral = "E-cigs on sale now. Buy them before we run out"

# Determine the sentiment of the text.
# It will show you the amount of negative, neutral, positive, and overall (compound)
pos_scores = sentiment_analyzer.polarity_scores(text_pos)
neg_scores = sentiment_analyzer.polarity_scores(text_neg)
neutral_scores = sentiment_analyzer.polarity_scores(text_neutral)

# print the scores
print("pos_score = ", pos_scores)
print("neg_score = ", neg_scores)
print("neutral_score = ", neutral_scores)
print("\n")


# Show the sentiment of individual words in sentences
words = nltk.word_tokenize(text_pos)
for word in words:
    scores = sentiment_analyzer.polarity_scores(word)
    print("word: " + str(word) + ", scores: " + str(scores))