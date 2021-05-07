import csv
import json
import re

import preprocessor as p
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import CSV_COLUMNS, SHORT_WORD_LENGTH
from utils.data_set import CMUTweetTagger
from utils.data_set.pcfg_utils import (extract_pcfg_trees_from_tweet,
                                       extract_rules_from_pcfg_trees)

tweet_source_dict = dict()

# insert the name of the tagged person insted of this profile name


def change_user_mentions(text: str, index1: int, index2: int, name: str):
    new_text = "".join((text[:index1], name, text[index2:]))
    return new_text

# replace the sub string between index1 and index2 with "relpace_with" text


def string_replace(text: str, index1: int, index2: int, replace_with: str):
    new_text = "".join((text[:index1], replace_with, text[index2:]))
    return new_text

# handles each entity of the tweet (user_mentions,url,hastahs,symbols)


def replace_user_mentions(user_mentions_array: list, text: str):
    ret_text = text
    sorted_user_mentions = sorted(
        user_mentions_array, key=lambda x: x['indices'][0], reverse=True)
    for user_mention in sorted_user_mentions:
        indices = user_mention['indices']
        index1 = indices[0]
        index2 = indices[1]
        text_length = len(text)
        if(index2 > text_length):
            continue
        ret_text = change_user_mentions(
            ret_text, index1, index2, user_mention['name'])
    return ret_text


# remove emojis from the text
def remove_emojis(text):
    p.set_options('emojis')
    no_emojis_text = p.clean(text)
    return no_emojis_text

# remove urls and hashtags from the tweet


def fix_tweet(tweet_text):
    p.set_options('urls', 'hashtags')
    clean_test = p.clean(tweet_text)
    clean_test = clean_test.replace("&amp;", "&")
    return clean_test

# remove url and hastags and replaces user mentions


def get_fixed_text_from_tweet(tweet: str):

    full_text = tweet.full_text
    user_mentions = tweet.entities['user_mentions']
    without_user_mentions = replace_user_mentions(user_mentions, full_text)
    clean_text = fix_tweet(without_user_mentions)
    return clean_text


def write_file_headers(file_writer):
    headers = list(CSV_COLUMNS.values())
    file_writer.writerow(headers)
    return

def text_to_one_line(text):
    lines = text.splitlines()
    one_line_text = ''.join(lines)
    return one_line_text

# get the final csv row data of the tweet

def get_tweet_row(decoded_text: str, encoded_text: str, tweet: object):
    lines = encoded_text.splitlines()
    one_line_text = ''.join(lines)
    # 0 -> fixed tweet
    tweet_data = [one_line_text]
    # 1 -> tweet length
    chars_length = len(one_line_text)
    tweet_data.append(chars_length)
    # 2- -> number of words
    words = encoded_text.split()
    tweet_data.append(len(words))
    # 3 -> number of lines
    tweet_data.append(len(lines))
    # 4 -> number of short words
    short_words = [x for x in words if len(x) < SHORT_WORD_LENGTH]
    tweet_data.append(len(short_words))
    # 5 -> upload hour
    if(tweet.created_at):
        tweet_data.append(tweet.created_at.hour)
    else:
        tweet_data.append(-1)
    # 6 -> tweet source
    if(tweet.source):
        tweet_source_enum = -1
        if(tweet.source in tweet_source_dict):
            tweet_source_enum = tweet_source_dict[tweet.source]
        else:
            tweet_source_enum = len(tweet_source_dict.keys()) + 1
            tweet_source_dict[tweet.source] = tweet_source_enum
        tweet_data.append(tweet_source_enum)
    else:
        tweet_data.append(-1)
    if(tweet.entities):
        # 7 -> included media (?)
        if('media' in tweet.entities and len(tweet.entities['media']) > 0):
            tweet_data.append("1")
        else:
            tweet_data.append("0")

        # 8 -> user mentions
        if('user_mentions' in tweet.entities):
            tweet_data.append(str(len(tweet.entities['user_mentions'])))
        else:
            tweet_data.append("0")

        # 9 -> number of urls
        if('urls' in tweet.entities):
            tweet_data.append(str(len(tweet.entities['urls'])))
        else:
            tweet_data.append("0")

        # 10 -> number of hashtags
        if('hashtags' in tweet.entities):
            tweet_data.append(str(len(tweet.entities['hashtags'])))
        else:
            tweet_data.append("0")

    # 11 -> emojis
    emojis_data = find_emojis(decoded_text)
    tweet_data.append(emojis_data)

    # 12 -> pcfg rules
    pcfg_trees = extract_pcfg_trees_from_tweet(one_line_text)
    rules_from_pcfg = extract_rules_from_pcfg_trees(pcfg_trees)
    rules_str = json.dumps(rules_from_pcfg)
    tweet_data.append(rules_str)

    # 13 -> sentiment score
    sid = SentimentIntensityAnalyzer()
    res = sid.polarity_scores(encoded_text)
    tweet_data.append(res['neg'])
    tweet_data.append(res['neu'])
    tweet_data.append(res['pos'])

    # 14 -> POS tagging (written at the end)

    return tweet_data

# find the emojis in the text and return a summary version of them
def find_emojis(text: str):
    emoji_pattern = re.compile('[\U0001F300-\U0001F64F]')
    emoji_counter = dict()
    for m in re.finditer(emoji_pattern, text):
        emoji = m.group(0)
        if(emoji in emoji_counter):
            emoji_counter[emoji] += 1
        else:
            emoji_counter[emoji] = 1
    emoji_counter_text = ""
    first_flag = True
    for key, value in emoji_counter.items():
        if(first_flag):
            first_flag = False
        else:
            emoji_counter_text += ","
        emoji_counter_text += str(key) + ":" + str(value)
    return emoji_counter_text


def get_POS_tags_and_write_to_file(tweets_list,user_data_list,output_path):
    try:
        # for idx,tweet in enumerate(tweets_list):
        #     print(idx)
        tweets_POS = CMUTweetTagger.runtagger_parse(tweets_list)
        with open(output_path, mode='w+', newline='', encoding='utf-8') as user_file:
            csv_writer = csv.writer(
                user_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            assert len(tweets_POS) == len(user_data_list)        
            write_file_headers(csv_writer)
            for idx, row in enumerate(user_data_list):
                tweet_pos = tweets_POS[idx]
                POS_str = json.dumps(tweet_pos)
                row.append(POS_str)
                csv_writer.writerow(row)
    except Exception as e: 
        raise Exception(e)
