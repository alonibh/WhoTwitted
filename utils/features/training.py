import json
from heapq import nlargest

import nltk
import re
from nltk.util import ngrams

import config
from utils.data_set.pcfg_utils import (combine_rules,
                                       convert_top_rules_to_vector,
                                       get_most_frequent_pcfg_rules,
                                       get_rules_vectorized)


class training():
    _most_frequent_1_gram = []
    _most_frequent_2_gram = []
    _most_frequent_pcfg_rules = list()

    def __init__(self):
        nltk.download('punkt')

    # extract num_of_top_values top values (by occournces) from ngram
    def extract_most_frequent_ngrams(self, tweets: list, N: int, num_of_top_values: int):
        ngrams_occurrences = dict()
        for tweet in tweets:
            tweet = tweet.lower()
            tweet = re.sub(r'[^a-zA-Z0-9\'\s]', ' ', tweet)
            tokens = [token for token in tweet.split(" ") if token != ""]
            ngram_list = list(ngrams(tokens, N))
            for ngram in ngram_list:
                ngram = ' '.join(ngram)
                if(ngram.lower() in config.FREQUANT_WORDS):
                    continue
                if ngrams_occurrences.get(ngram) is None:
                    ngrams_occurrences[ngram] = 1
                else:
                    ngrams_occurrences[ngram] += 1
        top_ngrams = nlargest(
            num_of_top_values, ngrams_occurrences, key=ngrams_occurrences.get)
        return top_ngrams

    # returns accournces for each N-gram in the tweet
    def extract_ngrams_from_tweet(self, tweet: str, N: int):
        ngrams_occurrences = dict()
        tweet = tweet.lower()
        tweet = re.sub(r'[^a-zA-Z0-9\'\s]', ' ', tweet)
        tokens = [token for token in tweet.split(" ") if token != ""]
        ngram_list = list(ngrams(tokens, N))
        for ngram in ngram_list:
            ngram = ' '.join(ngram)
            if ngrams_occurrences.get(ngram) is None:
                ngrams_occurrences[ngram] = 1
            else:
                ngrams_occurrences[ngram] += 1

        return ngrams_occurrences

    # number of emojis by the emoji str
    def get_num_of_emojis(self, emoji_str):
        emoji_dict = self.find_emoji_dict(emoji_str)
        num_of_emojis = 0
        for value in emoji_dict.values():
            num_of_emojis += value
        return num_of_emojis

    # full emoji vector with 1 and 0 by the emoji str
    def create_emoji_vec(self, emoji_str: str):
        emoji_vectorization = []
        emoji_dict = self.find_emoji_dict(emoji_str)
        if(len(emoji_dict.keys()) == 0):
            return [0] * len(self.full_emoji_list)
        for emoji in self.full_emoji_list:
            if emoji in emoji_dict.keys():
                emoji_vectorization.append(emoji_dict[emoji])
            else:
                emoji_vectorization.append(0)
        return emoji_vectorization

    # create the vector by ngram
    def create_ngram_vec(self, N: int, tweet: str):
        ngrams = dict()
        if N == 1:
            ngrams = self._most_frequent_1_gram
        elif N == 2:
            ngrams = self._most_frequent_2_gram

        vec = [0] * len(ngrams)

        tweet_ngrams = self.extract_ngrams_from_tweet(tweet, N)

        for idx, y_ngram in enumerate(ngrams):
            if y_ngram in tweet_ngrams.keys():
                vec[idx] = tweet_ngrams[y_ngram]
        return vec

    # read all the pcfg rules for each tweet and create the most frequent
    def extract_pcfg_top_rules(self, pcfg_rules_by_author: list):
        frequent_rules_by_author = dict()
        for author, pcfg_rules in pcfg_rules_by_author.items():
            rules = dict()
            for tweet_rules in pcfg_rules:
                tweet_rules_dict = json.loads(tweet_rules)
                rules = combine_rules(rules, tweet_rules_dict)
            most_frequent_rules = get_most_frequent_pcfg_rules(
                rules, config.TOP_PCFG_RULES)
            frequent_rules_by_author[author] = most_frequent_rules

        top_rules = dict()
        for _, frequent_rules_tuples in frequent_rules_by_author.items():
            for rule in frequent_rules_tuples:
                a = rule[0]
                b = rule[1]
                if(not a in top_rules.keys()):
                    top_rules[a] = [b]
                elif(not b in top_rules[a]):
                    top_rules[a].append(b)
        return top_rules

    def get_POS_tags_vector(self, POS_dict: dict, number_of_words: int):
        tags_vector = []
        for tag in config.POS_FREQUENT_TAGS:
            if tag in POS_dict.keys():
                occourances_percent = round(POS_dict[tag]/number_of_words, 2)
                tags_vector.append(occourances_percent)
            else:
                tags_vector.append(0)
        return tags_vector

    def get_X_headers(self):
        headers = []
        if config.USE_LEXICAL_FEATURES:
            headers.append("words_number")
            headers.append("number_of_lines")
            headers.append("number_of_short_words")
            headers.append("tweet_length_chars")
            headers.append("number_of_emojis")

        if config.USE_TWITTER_FEATURES:
            headers.append("tweet_hour")
            headers.append("tweet_source_code")
            headers.append("tweet_include_media")
            headers.append("number_of_tags")
            headers.append("number_of_links")
            headers.append("number_of_hashtags")

        if config.USE_NGRAM_FEATURES:
            headers += ["emoji #" + str(idx + 1) + " - " + emoji for idx,
                        emoji in enumerate(self.full_emoji_list)]  # emojis
            headers += ["ngram_1 #" + str(idx + 1) + " - " + ngram_1 for idx,
                        ngram_1 in enumerate(self._most_frequent_1_gram)]  # ngram 1
            headers += ["ngram_2 #" + str(idx + 1) + " - " + ngram_2 for idx,
                        ngram_2 in enumerate(self._most_frequent_2_gram)]  # ngram 2

        if config.USE_NLP_FEATURES:
            headers += ["rule #" + str(idx + 1) + " - " + rule for idx,
                        rule in enumerate(self.all_top_rules_vector)]
            headers.append("Negativity Score")
            headers.append("Neutrality Score")
            headers.append("Positivity Score")
            headers += ["POS tag - '"+ tag +
                        "'" for tag in config.POS_FREQUENT_TAGS]

        return headers

    def vectorize(self, tweet_data: dict):
        vec = []

        if config.USE_LEXICAL_FEATURES:
            # 1 -> number of words
            vec.append(int(tweet_data[config.NUMBER_OF_WORDS_COLUMN_NAME]))
            # 2 -> number of lines
            vec.append(int(tweet_data[config.NUMBER_OF_LINES_COLUMN_NAME]))
            # 3 -> number of short words
            vec.append(
                int(tweet_data[config.NUMBER_OF_SHORT_WORDS_COLUMN_NAME]))
            # 4 -> tweet length (in chars)
            vec.append(int(tweet_data[config.TWEET_LENGTH_COLUMN_NAME]))
            # 5 -> number of emojis
            vec.append(self.get_num_of_emojis(
                tweet_data[config.EMOJIS_COLUMN_NAME]))

        if config.USE_TWITTER_FEATURES:
            # 6 -> hour of tweet
            vec.append(int(tweet_data[config.TWEET_HOUR_COLUMN_NAME]))
            # 7 -> tweet source code
            vec.append(int(tweet_data[config.TWEET_SOURCE_COLUMN_NAME]))
            # 8 -> does tweet include media
            vec.append(int(tweet_data[config.INCLUDE_MEDIA_COLUMN_NAME]))
            # 9 -> number of tags
            vec.append(int(tweet_data[config.NUMBER_OF_TAGS_COLUMN_NAME]))
            # 10 -> number of links
            vec.append(int(tweet_data[config.NUMBER_OF_LINKS_COLUMN_NAME]))
            # 11 -> number of hashtags
            vec.append(int(tweet_data[config.NUMBER_OF_HASHTAGS_COLUMN_NAME]))

        if config.USE_NGRAM_FEATURES:
            # 12 -> emoji vector (number of uniq emojis)
            vec += self.create_emoji_vec(tweet_data[config.EMOJIS_COLUMN_NAME])
            # 13 -> 1 ngram vector (config.top_ngram * number of tweeters)
            vec += self.create_ngram_vec(1,
                                         tweet_data[config.TWEET_COLUMN_NAME])
            # 14 -> 2 ngram vector (config.top_ngram * number of tweeters)
            vec += self.create_ngram_vec(2,
                                         tweet_data[config.TWEET_COLUMN_NAME])

        if config.USE_NLP_FEATURES:
            # 15 -> pcfg rules vector (unique pcfg rules)
            rules_dict = json.loads(
                tweet_data[config.PCFG_RULES_COLUMN_NAME])
            vec += get_rules_vectorized(rules_dict, self.all_top_rules_vector)

            # 16 -> sentiment score
            vec.append(float(tweet_data[config.NEGATIVITY_SCORE_COLUMN_NAME]))
            vec.append(float(tweet_data[config.NEUTRALITY_SCORE_COLUMN_NAME]))
            vec.append(float(tweet_data[config.POSITIVITY_SCORE_COLUMN_NAME]))

            # 17 -> POS tagging
            POS_dict = json.loads(
                tweet_data[config.PART_OF_SPEECH_COLUMN_NAME])
            vec += self.get_POS_tags_vector(POS_dict, int(
                tweet_data[config.NUMBER_OF_WORDS_COLUMN_NAME]))

        return vec

    # get the string from the csv and retrieve the list of the emojis that got used in the tweet
    def find_emoji_dict(self, emoji_str):
        emoji_obj = dict()
        if(emoji_str == ""):
            return emoji_obj
        comma_splitted = emoji_str.split(",")
        for obj in comma_splitted:
            emoji_code = obj.split(":")[0]
            emoji_occurrences = int(obj.split(":")[1])
            emoji_obj[emoji_code] = emoji_occurrences
        return emoji_obj

    # retrieve emoji data from each tweet and creates list of all the emojis used in the tweets
    def get_all_emojis_vec(self, all_tweets_data):
        emoji_vec = []
        emojis_data = [x[config.EMOJIS_COLUMN_NAME] for x in all_tweets_data]
        for emoji_data in emojis_data:
            if(emoji_data != ""):
                emoji_dict = self.find_emoji_dict(emoji_data)
                for emoji in emoji_dict.keys():
                    if not emoji in emoji_vec:
                        emoji_vec.append(emoji)

        return emoji_vec

    # initializing emoji,pcfg and ngram global variables
    def initialize_self_variables(self, all_tweets_data, y):

        tweets_by_author = dict()
        pcfg_rules_by_author = dict()
        index = 0
        for tweet_data in all_tweets_data:
            author_name = y[index]
            if tweets_by_author.get(author_name) is None:
                tweets_by_author[author_name] = list()
            if pcfg_rules_by_author.get(author_name) is None:
                pcfg_rules_by_author[author_name] = list()
            tweets_by_author[author_name].append(
                tweet_data[config.TWEET_COLUMN_NAME])
            pcfg_rules_by_author[author_name].append(
                tweet_data[config.CSV_COLUMNS['pcfg_rules']])
            index += 1

        ngram_1_list = []
        ngram_2_list = []

        for _, tweets in tweets_by_author.items():
            ngram_1_list += (self.extract_most_frequent_ngrams(tweets,
                                                               1, config.TOP_NGRAM))
            ngram_2_list += (self.extract_most_frequent_ngrams(tweets,
                                                               2, config.TOP_NGRAM))

        self._most_frequent_1_gram = list(dict.fromkeys(ngram_1_list))
        self._most_frequent_2_gram = list(dict.fromkeys(ngram_2_list))

        self.full_emoji_list = self.get_all_emojis_vec(all_tweets_data)

        top_rules_dict = self.extract_pcfg_top_rules(pcfg_rules_by_author)
        self.all_top_rules_vector = convert_top_rules_to_vector(top_rules_dict)

        self.x_headers = self.get_X_headers()

    # recieves tweets data and returns vector for each tweet data
    def vectorize_all_data(self, all_tweets_data: list, y: list):
        self.initialize_self_variables(all_tweets_data, y)
        # vectorizing the data
        print("Starting vectorization")
        X = []
        for _, tweet_data in enumerate(all_tweets_data):
            X.append(self.vectorize(tweet_data))

        print("Finished vectorization")
        return X

    # receives X and Y and fit the data by the classifier
    def train(self, X: list, Y: list, classifier):
        assert len(X) == len(Y)

        model = classifier
        model.fit(X, Y)

        return model
