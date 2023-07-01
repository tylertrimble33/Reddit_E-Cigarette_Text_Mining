import os
import pandas as pd

data_dir = 'data'
input_file = 'labeled_processed_reddit_data_sample'
df = pd.read_csv(os.path.join(data_dir, input_file), sep='|', on_bad_lines='warn')

very_bad = 0
bad = 0
neutral = 0
good = 0
very_good = 0

for score in df['sentiment_score']:
    if score <= -0.75:
        very_bad += 1
    if score <= -0.1:
        bad += 1
    if score >= 0.1:
        good += 1
    if score >= 0.75:
        very_good += 1
    else:
        neutral += 1

print("Number of scores: ", df['sentiment_score'].count())
print("Very Bad: ", very_bad)
print("Bad: ", bad)
print("Neutral: ", neutral)
print("Good: ", good)
print("Very Good: ", very_good)
