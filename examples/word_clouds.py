import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import re

# load the data
data_dir = '../data'
input_file = 'labeled_processed_sample_data'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='|', on_bad_lines='warn')
tweets = df['processed_strings'].tolist()
numClusters = max(df['cluster'])+1

# --------------- Word Cloud
# create a word cloud for each cluster
for clusterNum in range(numClusters):

    # create a string of all the tweets in this cluster
    cluster_samples = df.loc[df['cluster'] == clusterNum]
    cluster_text = ''.join(cluster_samples['processed_strings'])

    # process text to make word clouds prettier/more informative
    # but, I should have just removed these words as stop words in the beginning
    cluster_text = re.sub(' vape ', '', cluster_text)
    cluster_text = re.sub(' vaping ', '', cluster_text)
    cluster_text = re.sub(' ecig ', '', cluster_text)

    # generate the word cloud
    wc = WordCloud(width=1600, height=800, collocations=False, max_words=30).generate(cluster_text)
    default_colors = wc.to_array()
    plt.title("Custom colors")
    plt.imshow(wc.recolor(random_state=3))
    wc.to_file("wordcloud_c" + str(clusterNum) + ".png")
