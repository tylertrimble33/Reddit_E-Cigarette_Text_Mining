import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.preprocessing


# Code to plot a dendrogram, directly from:
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
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


# Load our cleaned data
data_dir = '../data'
input_file = 'processed_sample_data'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn')
tweets = df['processed_strings'].tolist()

#vectorize the data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(tweets)


## Agglomerative Clustering
# Initialize an agglomerative clustering model
# Cosine distance is preferred for most text-processing tasks
# To mimic the procedure described in the slides, linkage = 'average'. Other options may work well too.
# See the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
# Need to compute_distances to plot the dendrogram
# A distance_threshold could be used to prevent very dislike clusters from joining. Since cosine distance is always
#   between 0 and 1, you can experiment with a few, and/or analyze the dendrogram and pick a good one
agglomerative_model = AgglomerativeClustering(metric='cosine', linkage='average', compute_distances=True, distance_threshold=None)
# Fit the clustering model. Need to convert the sparse X_tfidf matrix into a dense one
#  using the toarray() function
agglomerative_model.fit(X_tfidf.toarray())
# the number of leaves correspond to the number of data points
print("n_leaves_ = ", agglomerative_model.n_leaves_)
# If you don't specify a distance_threshold, the number of clusters will be 2, since it will stop just before
#  merging the last 2 clusters. Therefore, the distance threshold could be a way to gauge an optimum number of clusters
print("n_clusters_ = ", agglomerative_model.n_clusters_)
# You can print the labels=the cluster they belong to, of each point in the dataset.
print("labels_ = ", agglomerative_model.labels_)
# plot the full dendrogram, but this can be overwhelming and uninformative
#plot_dendrogram(agglomerative_model)
# Instead, plot just the top p=2 levels of the dendrogram
#    (for datasets larger than the example one you will probably want to plot more than p=2)
plot_dendrogram(agglomerative_model, truncate_mode="level", p=2)






## K-Means Clustering
# Create a K-means model.
# The K-means algorithm will repeat n_init number of times. The best result of clustering will be used. Since
#   the initial cluster centroids are random, the quality of the clusters can vary a lot depending on these
#   random seeds, especially with sparse data. Therefore, trying multiple times is highly recommended.
#   The best set of generated clusters is determined by "inertia" = the mean inter-cluster distance
#   (see documentation) https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
print("X_tfidf.shape = ", X_tfidf.shape)
kmeans_model = KMeans(n_clusters=5, n_init=20)

# Scikit-learn's implementation of K-means doesn't allow cosine distance. It may be worthwhile to explore other
#   clustering algorithms. See: https://scikit-learn.org/stable/modules/clustering.html
# The major benefit of cosine distance is that it is scale invariant, so to mimic this I normalize the X_tfidf
#   matrix. This isn't a perfect solution, because there is no guarantee the cluster centroids used during
#   the clustering process will be of unit length.
# I specify axis=1, because I want to normalize each row. It is important to check that you selected
#   the correct axis and that normalization worked as expected. So, I calculate the length of each
#   normalized vector as well as the number of normalized vectors. The lengths should all be 1 (they
#   should all be unit length vectors), and there should be 1 length output for each sample
# Always check to make sure your code is doing what is expected, especially when using other packages.
#   these packages are powerful, but also easy to misuse
X_norm = sklearn.preprocessing.normalize(X_tfidf, axis=1)
# Calculate the lengths and convert X_norm to an array rather than a sparse matrix.
#   I was getting SIGKILL errors when using a sparse matrix with np.linalg.norm
#   However, leaving X_norm as a sparse matrix for kmeans_model.fit makes it WAY faster
#   so, we don't want to keep it sparse there.
lengths = np.linalg.norm(X_norm.toarray(), axis=1)
print ("X_norm sums = ", lengths)
print ("shape of X_norm sums = ", lengths.shape)
kmeans_model.fit(X_norm)

# Obtain the predicted labels - these are the clusters each tweet maps to
labels = kmeans_model.labels_
print ("labels = ", labels)

# Output the number of samples per cluster. We don't want any clusters to be too small
# Code from: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
#   It's worth a read
cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
print(f"Number of elements asigned to each cluster: {cluster_sizes}")

# Check the quality of these clusters
print("Mean inter_cluster distance = ", kmeans_model.inertia_)

# TODO - You should also calculate and output the intra-cluster distance. You can access the cluster centroids with
#  kmeans_model.cluster_centers_ - then, calculate the average distance between each center. Try multiple values of
#  k and pick the best using these values and manual analysis

# map the labels back to the dataframe and save it
df['cluster'] = labels
df.to_csv(os.path.join(data_dir, "labeled_" + str(input_file)), sep="|")
print("Done")
