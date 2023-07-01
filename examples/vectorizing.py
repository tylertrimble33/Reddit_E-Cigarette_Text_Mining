import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the cleaned data
data_dir = '../data'
input_file = 'processed_sample_data'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn')

# We are going to convert our documents (tweet) into a vector
# So, we need either a list of strings (each string is a tweet)
tweets = df['processed_strings'].tolist()
print("number of Tweets = ", len(tweets))
#print("sentences = ", sentences)

# Or, a list of lists, where each sublist is a list of tokens
# But, recall, we already used the tokenizer in the text-cleaning step. Since there is a chance the tokenizer
# tokenized better than just space-separating, we may want to use the tokenizer, but remember we had to
# recombine numbers %%%, so I will use the space separating.
#sentences = df['tokens'].tolist()
#print("sentences = ", sentences)

# This is a bag -of-words vectorizer
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(tweets)
vocab = vectorizer.get_feature_names_out()
# Note, X is a csr_matrix (compressed row storage matrix = a sparse matrix)'
print("X_bow.shape = ", X_bow.shape)
print("bow vocab = ", vocab)

# For TFIDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(tweets)
vocab = vectorizer.get_feature_names_out()
print("X_tfidf.shape = ", X_tfidf.shape)
print("tf-idf vocab = ", vocab)

# Check for any zero vectors which will mess up using cosine distance in clustering
#   These should have been removed during text cleaning (it implies there are 0 words), but its a good idea to check
sums = X_tfidf.sum(axis=1)
print("sums.shape = ", sums.shape)
for i, sum in enumerate(sums):
    if sum == 0:
        print("sum " + str(i) + " = 0, tweet = " + str(tweets[i]))