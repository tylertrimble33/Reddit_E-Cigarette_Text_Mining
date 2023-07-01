import pandas as pd
import nltk
import re
import os


def remove_stop_words(df):
    # tokenize and remove stop words
    tokenizer = nltk.tokenize.TweetTokenizer(False, False, True)
    # nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    vape_words = ['vape', 'vaping', 'ecig']
    stopwords.extend(vape_words)

    df['tokens'] = df['string'].apply(tokenizer.tokenize)
    df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])
    # join tokens with a space
    df['processed_strings'] = df['tokens'].apply(lambda x: str.join(' ', x))

    # re-merge percent signs (%'s). These were put in there to represent numbers, but the tokenizer splits them up
    # I will iteratively replace %% pairs (up to 100 %'s in a row) with non space separated %%'s
    # 100 is arbitrary, and I should probably check to see if it covers everything, but I don't think anyone typed more than 100 digits in a row(?)
    for num in reversed(range(20)):
        df['processed_strings'] = df['processed_strings'].apply(
            lambda x: re.sub('% %( %){' + str(num) + '}', '%%' + '%' * num,
                             x))  # create a regular expression to replace % % and any more % with the correct number of %'s <-- TODO, unpack this and show to them. Good example of complex regular expressions
    return df


# Input parameters
data_dir = 'data'
input_file = 'reddit_data_sample'

# Load the data
df = pd.read_csv(os.path.join(data_dir, input_file), sep='*', on_bad_lines='warn', encoding='ISO-8859-1')
print("data loaded df.shape: ", df.shape)   # Check shape of dataframe

# Data cleanup
df['string'] = df['comments'].apply(lambda x: str(x))   # Convert non UTF-8 characters to a string
df['string'] = df['string'].apply(lambda x: re.sub(r'\\n', ' ', x))     # replace new lines with space
df = df[~df['string'].str.contains(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', flags=re.IGNORECASE)]
print("Removed URLs: " + str(df.shape))
df['string'] = df['string'].apply(lambda x: re.sub(r"[^A-z0-9%@<>+']", ' ', x))    # replace characters that are not alphamumeric or %@<>+' with a space
df['string'] = df['string'].apply(lambda x: re.sub(r'[\'@<>+-]', '', x))    # remove special characters
df['string'] = df['string'].apply(lambda x: re.sub(r'amp', '', x))  # remove "amp"
df = remove_stop_words(df)  # remove stop words

# Remove empty tweets
df = df.loc[(df['processed_strings'] != '')]
print("Removed Empty Tweets: " + str(df.shape))

# Remove tweets with 2 or less words
df = df[df['processed_strings'].apply(lambda x: len(x.split(' ')) > 2)]
print("Removed Short Tweets: " + str(df.shape))

# Remove marijuana and cbd related posts
cbd_words = ['cbd', 'cannabidiol', 'marijuana', 'weed', 'cannabis', 'thc', 'hemp']
df = df[~df['processed_strings'].str.lower().str.contains('|'.join(cbd_words))]
print("Removed weed tweets: " + str(df.shape))

# Remove deleted rows
df = df[~(df['processed_strings'] == '[ deleted ]')]
print("Removed deleted posts: " + str(df.shape))

# Remove rows with no words
df = df[df['processed_strings'].str.contains(r'\b', regex=True)]
print('Removed no words: ' + str(df.shape))

# Remove unnecessary columns
# df.drop(columns=['string'], inplace=True)
df.drop(columns=['tokens'], inplace=True)
df.drop(columns=['comments'], inplace=True)
df.drop(columns=['ids'], inplace=True)
df.drop(columns=['titles'], inplace=True)

# HARDCODED SOLUTION TO A PROBLEM WITH DATASET... TODO: FIX!!!
df = df[~(df['processed_strings'] == 'l k e')]

# Output the processed data frame
df.to_csv(os.path.join(data_dir, "processed_" + str(input_file)), sep="*")
print("Done")
