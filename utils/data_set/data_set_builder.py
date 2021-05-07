import nltk
import tweepy

import config
from auth import twitter_credentials

from .data_set_utils import (get_fixed_text_from_tweet,
                             get_POS_tags_and_write_to_file, get_tweet_row,
                             remove_emojis, text_to_one_line)


def create_dataset(input_file_path: str, output_dir: str):
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    # OAuth process, using the keys and tokens
    auth = tweepy.OAuthHandler(
        twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
    auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                          twitter_credentials.ACCESS_TOKEN_SECRET)

    # Creation of the actual interface, using authentication
    max_tweets = config.MAX_USER_TWEETS
    api = tweepy.API(auth)
    with open(input_file_path, "r") as lines:
        for line in lines:
            user = api.get_user(line)
            user_tweets = tweepy.Cursor(
                api.user_timeline, screen_name=line, tweet_mode="extended", count=100)
            user_name = user.name
            user_name_tweets_list = []
            all_user_rows = []
            valid_messages = 0
            for idx, tweet in enumerate(user_tweets.items()):
                if(valid_messages == max_tweets):
                    break
                if(hasattr(tweet, 'retweeted_status') or (tweet.lang != "en" and tweet.lang != 'und')):
                    print(user_name + " - #" + str(idx) + " - skipped")
                    continue
                # remove entities from the tweet (with emojis)
                fixed_text_with_emojis = get_fixed_text_from_tweet(tweet)
                try:
                    no_emojis_text = remove_emojis(fixed_text_with_emojis)
                    no_emojis_text = no_emojis_text.encode(
                        'ascii', 'ignore').decode("utf-8")
                    user_name_tweets_list.append(
                        text_to_one_line(no_emojis_text))
                    tweet_line = get_tweet_row(
                        fixed_text_with_emojis, no_emojis_text, tweet)
                    all_user_rows.append(tweet_line)
                    valid_messages += 1
                    print(user_name + " - #" + str(idx) + " - success")
                except Exception as e:
                    print(e)
                    print(user_name + " - #" + str(idx) + " - FAILED")
            # adding the pos taggers
            output_path = output_dir / (user_name+".csv")
            get_POS_tags_and_write_to_file(
                user_name_tweets_list, all_user_rows, output_path)
            print("finished writing to file - " + user_name)
            print("*************************")
