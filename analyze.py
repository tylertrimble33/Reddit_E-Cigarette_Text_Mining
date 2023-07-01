import pandas as pd
import os

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from wordcloud import WordCloud
import re

# Load the cleaned data
data_dir = 'data'
input_file = 'processed_reddit_data_sample'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn')

# Add each post into a list
processed_posts = df['processed_strings'].tolist()
print("number of Tweets = ", len(processed_posts))

# Vectorization
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(processed_posts)
vocab = vectorizer.get_feature_names_out()
print("X_bow.shape = ", X_bow.shape)  # Note, X is a csr_matrix (compressed row storage matrix = a sparse matrix)'
print("bow vocab = ", vocab)

# For TFIDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(processed_posts)
vocab = vectorizer.get_feature_names_out()
print("X_tfidf.shape = ", X_tfidf.shape)
print("tf-idf vocab = ", vocab)

# Check for zero vectors
sums = X_tfidf.sum(axis=1)
print("sums.shape = ", sums.shape)
for i, sum in enumerate(sums):
    if sum == 0:
        print("sum " + str(i) + " = 0, tweet = " + str(processed_posts[i]))

# Sentiment Analysis
sentiment_analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['processed_strings'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])


# Clustering
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.title("Hierarchical Clustering Dendrogram")

    dendrogram(linkage_matrix, **kwargs)
    # plot the top three levels of the dendrogram
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()


# Agglomerative Clustering
agglomerative_model = AgglomerativeClustering(metric='cosine', linkage='average', compute_distances=True,
                                              distance_threshold=None)
agglomerative_model.fit(X_bow.toarray())
print("n_leaves_ = ", agglomerative_model.n_leaves_)
print("n_clusters_ = ", agglomerative_model.n_clusters_)
print("labels_ = ", agglomerative_model.labels_)
plot_dendrogram(agglomerative_model, truncate_mode="level", p=2)

# K-Means Clustering
print("X_tfidf.shape = ", X_bow.shape)
kmeans_model = KMeans(n_clusters=17, n_init=20)
X_norm = sklearn.preprocessing.normalize(X_bow, axis=1)
lengths = np.linalg.norm(X_norm.toarray(), axis=1)
print("X_norm sums = ", lengths)
print("shape of X_norm sums = ", lengths.shape)
kmeans_model.fit(X_norm)

# Obtain the predicted labels - these are the clusters each tweet maps to
labels = kmeans_model.labels_
print("labels = ", labels)

# Output the number of samples per cluster
cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
print(f"Number of elements asigned to each cluster: {cluster_sizes}")

# Check the quality of these clusters
print("Mean inter_cluster distance = ", kmeans_model.inertia_)

# Calculate average inter-cluster distance
avg_intercluster_dist = pairwise_distances(kmeans_model.cluster_centers_).mean()
print("Average intercluster distance: ", avg_intercluster_dist)

# Map the labels back to the dataframe and save it
df['cluster'] = labels
df.to_csv(os.path.join(data_dir, "labeled_" + str(input_file)), sep="|")


# Topic Modeling
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


# Create an instance of LDA and fit the LDA model to your data
lda_model = LatentDirichletAllocation(n_components=7)
lda_model.fit(X_tfidf)

# Obtain the topic-word distributions - will explain the amount of each word in each topic
topic_word_distributions = lda_model.components_

# Obtain the document-topic distributions -- will explain how much of each topic is in each document
document_topic_distributions = lda_model.transform(X_tfidf)
plot_topics(lda_model, vocab, 7, '7 Topics Found using LDA')

# Word Clouds
# load the labeled data
data_dir = 'data'
input_file = 'labeled_processed_reddit_data_sample'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='|', on_bad_lines='warn')
processed_posts = df['processed_strings'].tolist()
numClusters = max(df['cluster']) + 1

# create a word cloud for each cluster
for clusterNum in range(numClusters):
    # create a string of all the tweets in this cluster
    cluster_samples = df.loc[df['cluster'] == clusterNum]
    cluster_text = ''.join(cluster_samples['processed_strings'])

    # process text to make word clouds prettier/more informative
    cluster_text = re.sub(' vape ', '', cluster_text)
    cluster_text = re.sub(' vaping ', '', cluster_text)
    cluster_text = re.sub(' ecig ', '', cluster_text)

    # generate the word cloud
    wc = WordCloud(width=1600, height=800, collocations=False, max_words=30).generate(cluster_text)
    default_colors = wc.to_array()
    plt.title("Custom colors")
    plt.imshow(wc.recolor(random_state=3))
    wc.to_file("wordcloud_c" + str(clusterNum) + ".png")

print("Analysis Complete")
