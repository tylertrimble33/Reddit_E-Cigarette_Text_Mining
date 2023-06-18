import praw
from praw.models import MoreComments
import pandas as pd

# Gets the top post IDs
def get_post_ids(subreddit_name, post_limit=None):
    posts = reddit.subreddit(subreddit_name).top(limit=post_limit)
    post_ids = [post.id for post in posts]
    return post_ids

# for getting the top level comments
#   This therefore won't grab whole discussions of comments.
#   See PRAW documentation if you want to do more: https://praw.readthedocs.io/en/stable/
def get_top_level_comments(submission, number_comments_to_get=500, replace_more_limit=None):
    count = 0
    comments = []
    # the submission.comments is a "comment_forest" which may contain
    #  "MoreComments" objects. These are equivalent to clicking the "more comments"
    #   button on Reddit. Each call to this is a server request, and we are rate_limited
    #   so, by setting limit=None we will not retrieve more comments. It may be good to
    #   retreive more though, so you can change this if desired
    #   See: https://praw.readthedocs.io/en/stable/tutorials/comments.html#the-replace-more-method
    submission.comments.replace_more(limit=replace_more_limit)

    # get the top n comments
    for top_level_comment in submission.comments:

        # skip a "more comment" object - needed if replace_more_limit isn't None
        #   because a "MoreComments" object may exist and they don't have a .body attribute
        if isinstance(top_level_comment, MoreComments):
            continue

        # append the comment body
        comments.append(top_level_comment.body)

        # get just the top comments
        count += 1
        if count == number_comments_to_get:
            break

    return comments

# You may want to get all the comments in a thread rather than just the top-level comments
#   e.g. all the replies and replies of replies.
# The following code will retrieve all comments using a breadth-first-search of the comment
#  forest. If you are only interested in the top one or two level, that code is
#  pretty easy to write. The PRAW documentation shows you how (look near the end).
#     https://praw.readthedocs.io/en/stable/tutorials/comments.html#the-replace-more-method
def get_all_level_comments(submission, replace_more_limit=None):
    # See get_top_level_comments for explanation of these steps
    comments = []
    submission.comments.replace_more(limit=replace_more_limit)
    for comment in submission.comments.list():
        if isinstance(comment, MoreComments):
            continue
        comments.append(comment.body)
    return comments


# instantiate an instance of the Reddit API
reddit = praw.Reddit(client_id='YOUR CLIENT ID',
                     client_secret='YOUR CLIENT SECRET',
                     user_agent='YOUR APP NAME')

print("API instance initialized")

# Hardcoded subreddit list - TODO - add more subreddits to this list. Find more and better ones
subreddit_list = ['electronic_cigarette']

# grab post ids in the specified subreddits
# TODO - you probably want to increase the post_limit from 10. Do you need any limit?
post_ids =[]
for subreddit_name in subreddit_list:
    post_ids.extend(get_post_ids(subreddit_name, post_limit=10))
print("got post ids")

# grab titles of the post-ids
post_titles = []
for id in post_ids:
    submission = reddit.submission(id=id)
    post_titles.append(submission.title)
print("got titles")

# grab comments for the post-ids
all_comments=[]
for id in post_ids:
    # retrieves the submission for the comment id
    # submission objects contain "comment forests". See documentation
    #    https://praw.readthedocs.io/en/stable/tutorials/comments.html
    submission = reddit.submission(id)

    # Get top level, or all comments for posts.
    #  Don't just blindly grab data. Think about what you need and how you will store and interpret it
    #  Are you interested in discussions or just top comments? Do you have too much data, or do you
    #   need more data? How will you group and parse discussions vs. top comments vs. titles?
    #post_comments = get_top_level_comments(submission, number_comments_to_get=10, replace_more_limit=None)
    post_comments = get_all_level_comments(submission, replace_more_limit=5)
    all_comments.append(post_comments)
print("got comments")

# Save the data to a pandas dataframe. The comments are a list of comments for each id/title
df = pd.DataFrame({'ids': post_ids, 'titles': post_titles, 'comments': all_comments})
df.to_csv('reddit_data', sep='|') # "|" may not be the best sep token, since maybe someone types that? you may want to use something more complex (e.g. |*|)

