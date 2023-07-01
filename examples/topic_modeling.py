import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# plot the topics. Code taken directly from scikit-learn's demo:
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_topics(model, feature_names, n_top_words, title):
    # Visualize the results
    # Plot top words
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


# Load our cleaned data
data_dir = '../data'
input_file = 'processed_reddit_data'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn')
tweets = df['processed_strings'].tolist()

#vectorize the data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(tweets)
vocab = vectorizer.get_feature_names_out()

# Create an instance of LDA
lda_model = LatentDirichletAllocation(n_components=10)

# Fit the LDA model to your data
lda_model.fit(X_tfidf)

# Obtain the topic-word distributions - will explain the amount of each word in each topic
topic_word_distributions = lda_model.components_
print ("topic_word_distributions = ", topic_word_distributions)

# Obtain the document-topic distributions -- will explain how much of each topic is in each document
document_topic_distributions = lda_model.transform(X_tfidf)
print ("document_topic_distributions = ", document_topic_distributions)

plot_topics(lda_model, vocab, 10, '10 Topics Found using LDA')
