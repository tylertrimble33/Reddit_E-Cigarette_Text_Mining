import pandas as pd
import nltk
import re
import os
from langdetect import detect

## Methods to determine the language of a tweet
#  This didn't work well for me, but I applied it after processing
#  and after stop-word removal, and on Tweets. Maybe it's useful for you?
def get_lang(x):
    if (x != ''):
        try:
            lang = detect(x)
            if (lang != 'en'):
                print("lang = " + str(lang) + ", text = " + str(x))
            return lang
        except:
            print(x)
            return 'NONE'
    else:
        print(x)
        return 'NONE'


## Method to remove stop words
def remove_stop_words(df):
    # tokenize and remove stop words
    # TODO: You may want to use a different tokenizer. This one is specific to Tweets.
    #       If text contains hashtags, emojies, etc.. then the Tweet tokenizer may be appropriate
    #       Otherwise, try a General English Tokenizer, like: words = nltk.word_tokenize(sentence)
    #       No need to instantiate a tokenizer object, but you may need to nltk.download('punkt')
    #       for it it work (see sentiment_analysis.py for an example)
    tokenizer = nltk.tokenize.TweetTokenizer(False, False, True)
    # nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    df['tokens'] = df['string'].apply(tokenizer.tokenize)
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])
    # join tokens with a space
    df['processed_strings'] = df['tokens'].apply(lambda x: str.join(' ', x))

    # re-merge percent signs (%'s). These were put in there to represent numbers, but the tokenizer splits them up
    # I will iteratively replace %% pairs (up to 100 %'s in a row) with non space separated %%'s
    # 100 is arbitrary, and I should probably check to see if it covers everything, but I don't think anyone typed more than 100 digits in a row(?)
    for num in reversed(range(20)):
        df['processed_strings'] = df['processed_strings'].apply(
            lambda x: re.sub('% %( %){' + str(num) + '}', '%%' + '%' * num, x)) # create a regular expression to replace % % and any more % with the correct number of %'s <-- TODO, unpack this and show to them. Good example of complex regular expressions
    return df


# Input parameters
data_dir = '../data'
input_file = 'small_example.txt'

## Load the data
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn', encoding='ISO-8859-1')
# Check the shape of the dataframe. This will indicate the number of samples and the number of columns
print ("data loaded df.shape = ", df.shape)

## Process the Text
# Replace NA in quoted text field with an empty string
df['quote_text_processed_no_hash'].fillna(value='', inplace=True)
# combine tweet text and quoted text into a single string which we will process
df['combined_tweets_processed'] = df.apply(lambda x: str(x['tweet_text_processed_no_hash']) + ' ' + str(x['quote_text_processed_no_hash']), axis=1)

# I could remove non UTF-8 characters, one method is to do something like this, but this is incomplete. There are alsu u00 characters
#  However, emojis are important so I decided to keep them
#df['string'] = df['combined_tweets_processed'].apply(lambda x: re.sub(r'[^\x00-\x7F]+', ' ', str(x)))

# Convert non UTF-8 characters to a string
df['string'] = df['combined_tweets_processed'].apply(lambda x: str(x))
# replace new lines with space (new line is usually \n, but may be \r\n or just \r)
df['string'] = df['string'].apply(lambda x: re.sub(r'\r?\n|\r', ' ', x))
# replace characters that are not alphamumeric or %@<>+' with a space (i.e. -,?!.*$) # I'm not sure I should do this, why replace $ with a space?
df['string'] = df['string'].apply(lambda x: re.sub(r'[^A-z0-9%@<>+\']', ' ', x))
# remove special characters (keep % cause it indicates a number) - Note: In earlier preprocessing, I replaced all digits with %
df['string'] = df['string'].apply(lambda x: re.sub(r'[\'@<>+-]', '', x))
# remove "amp"
df['string'] = df['string'].apply(lambda x: re.sub(r'amp', '', x))
#remove stop words
df = remove_stop_words(df)

#Note: This code doesn't break emojis up. Consider doing so in your code
#       Right now they are all combined - I found this via manual analysis of the
#       text/tokens (e.g. u0001f937u0001f3fbu200du2642sativa = 2 emojis and "sativa")

## filter the tweets
# remove any sources not from twitter
print("No Filters - " + str(df.shape))

# remove tweets with URLS (according the the 'tweet_has_url'field) since they may be ads
df = df.loc[(df['tweet_has_url'] == 'False') | (df['tweet_has_url'] == 'FALSE') | (df['tweet_has_url'] == False)]
print("Removed tweet_has_url - " + str(df.shape))
df = df.loc[(df['quote_has_url'] != 'True') & (df['quote_has_url'] != 'TRUE')]
print("Removed quote_has_url - " + str(df.shape))

# remove empty tweets
df = df.loc[(df['processed_strings'] != '')]
print("Removed Empty Tweets - " + str(df.shape))

# remove non-english tweets -- Turns out this doesn't work well for me (found via manual analysis)
#df['lang'] = df['processed_strings'].apply(lambda x: get_lang(x))
#df = df.loc[(df['lang'] == 'en')]
#df = df.drop(columns = ['lang'])
#print("Removed Non-English Tweets - "+str(df.shape))

# remove tweets with 2 or less words
df = df[df['processed_strings'].apply(lambda x: len(x.split(' ')) > 2)]
print("Removed Short Tweets - " + str(df.shape))

#Note: based on analysis, I need to filter out CBD and marijuana related tweets.
#      I wonder if that will be a problem with Reddit?

# remove unnecessary columns to prevent data from getting too big and make it easier to inspect
# You can specify just a lit of columns, but this makes it easy to see what is being dropped and kept
# If preprocessing takes a long time, be careful about dropping data. You don't want to have to run it again
df.drop(columns=['status_id'], inplace=True)
df.drop(columns=['created_at'], inplace=True)
df.drop(columns=['source'], inplace=True)
df.drop(columns=['user_id'], inplace=True)
df.drop(columns=['text'], inplace=True)
df.drop(columns=['quoted_text'], inplace=True)
df.drop(columns=['screen_name'], inplace=True)
df.drop(columns=['name'], inplace=True)
df.drop(columns=['location'], inplace=True)
df.drop(columns=['description'], inplace=True)
df.drop(columns=['url'], inplace=True)
df.drop(columns=['is_quote'], inplace=True)
df.drop(columns=['is_retweet'], inplace=True)
df.drop(columns=['tweet_has_url'], inplace=True)
df.drop(columns=['quote_has_url'], inplace=True)
df.drop(columns=['source_is_twitter'], inplace=True)
df.drop(columns=['tweet_text_processed'], inplace=True)
df.drop(columns=['tweet_text_processed_no_hash'], inplace=True)
df.drop(columns=['quote_text_processed'], inplace=True)
df.drop(columns=['quote_text_processed_no_hash'], inplace=True)
#df.drop(columns=['combined_text'], inplace=True)
df.drop(columns=['combined_text_no_hash'], inplace=True)
df.drop(columns=['hashtags_c'], inplace=True)
df.drop(columns=['combined_tweets_processed'], inplace=True)
df.drop(columns=['string'], inplace=True)
df.drop(columns=['tokens'], inplace=True)
#df.drop(columns=['processed_strings'], inplace=True)

####### Output the processed data frame
df.to_csv(os.path.join(data_dir, "processed_" + str(input_file)), sep="|")
print("Done")