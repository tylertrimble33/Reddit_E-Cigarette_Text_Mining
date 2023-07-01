import os.path

import praw
from praw.models import MoreComments
import pandas as pd


# Gets the top post IDs
def get_post_ids(subreddit_name, post_limit=None):
    posts = reddit.subreddit(subreddit_name).top(limit=post_limit)
    post_ids = [post.id for post in posts]
    return post_ids


# for getting the top level comments
def get_top_level_comments(submission, number_comments_to_get=500, replace_more_limit=None):
    count = 0
    comments = []
    submission.comments.replace_more(limit=replace_more_limit)

    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue

        comments.append(top_level_comment.body)
        count += 1
        if count == number_comments_to_get:
            break
        print("Comment number:")
        print(count)
    return comments


def get_all_level_comments(submission, replace_more_limit=None):
    comments = []
    submission.comments.replace_more(limit=replace_more_limit)
    for comment in submission.comments.list():
        if isinstance(comment, MoreComments):
            continue
        comments.append(comment.body)
    return comments


# instantiate an instance of the Reddit API
reddit = praw.Reddit(client_id='npuJbFPra4hW7GQAryG2EQ',
                     client_secret='OIT2DPMrffjrLorJgMwmwcrWXFnm8A',
                     user_agent='Data Mining Project')

print("API instance initialized")

# Hardcoded subreddit list
subreddit_list = ['Vaping', 'e_cigarette', 'electronic_cigarette', 'vapes']

# grab post ids in the specified subreddits
post_ids = []
for subreddit_name in subreddit_list:
    post_ids.extend(get_post_ids(subreddit_name, post_limit=250))
print("Got post ids")

# grab titles of the post-ids
post_titles = []
for id in post_ids:
    submission = reddit.submission(id=id)
    post_titles.append(submission.title)
    print("Got title: " + str(submission.title))
print("Got all titles")

# grab comments for the post-ids
post_ids_list = []
post_titles_list = []
comments_list = []

# grab comments for the post-ids
for id in post_ids:
    submission = reddit.submission(id)
    post_comments = get_top_level_comments(submission, number_comments_to_get=16, replace_more_limit=None)

    # Append the post ID and title for each comment
    post_ids_list.extend([id] * len(post_comments))
    post_titles_list.extend([submission.title] * len(post_comments))
    comments_list.extend(post_comments)
print("Got all comments")

# Create and save dataframe to csv
df = pd.DataFrame({'ids': post_ids_list, 'titles': post_titles_list, 'comments': comments_list})
data_dir = 'data'
output_name = 'reddit_data_sample'
df.to_csv(os.path.join(data_dir, output_name), sep='*', index=False)
