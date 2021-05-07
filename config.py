import os
from pathlib import Path

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

USE_NLP_FEATURES = True
USE_NGRAM_FEATURES = True
USE_LEXICAL_FEATURES = True
USE_TWITTER_FEATURES = True

NUMBER_OF_ITERATIONS_FOR_PREDICTING = 3
GROUP_SIZE = 10
NUMBER_OF_GROUPS = 100
VECTOR_SIZE = 5000

TMP_TWEET_FILE = "tmp_tweet.txt"
TMP_POS_FILE = "tmp_pos.txt"
MAX_USER_TWEETS = 5000
TEST_SIZE_RATIO = 0.25
TOP_PCFG_RULES = 50
TOP_NGRAM = 50
SHORT_WORD_LENGTH = 5

FREQUANT_WORDS = ['the', 'to', 'and', 'a', 'of', 'you', 'for', 'in', 'i',
                  'on', 'is', 'this', 'it', 'my', 'that', 'with', "be", 'we', 'so', 'your', 'at', 'are', 'all', 'our', 'love']
FREQUANT_PCFG_RULES = {'PP': ['IN#NP', 'TO#NP'], 'ROOT': ['S'], 'S': ['VP', 'NP#VP', 'NP#VP#.'],
                       'NP': ['NP#PP', 'PRP', 'DT#NN', 'NN', 'NNP', 'NNP#NNP', 'NNS', 'PRP$#NN'],
                       'VP': ['TO#VP', 'VB#NP'], 'ADVP': ['RB']}
POS_FREQUENT_TAGS = ['N', 'V', 'P', 'D', 'A', ',', '^' '!']

CLASSIFIERS = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=0.5, max_iter=6000),
    # AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis(),
    LogisticRegression(solver='liblinear', multi_class='auto'),
    ExtraTreesClassifier(),
    # MultinomialNB()
]

CSV_COLUMNS = dict(
    tweet='fixed tweet',

    tweet_length='tweet length',

    number_of_words='number of words',

    number_of_lines='number of lines',

    number_of_short_words='number of short words',

    tweet_hour='upload hour',

    tweet_source='tweet source code',

    include_media='included media(?)',

    number_of_tags='number of user mentions',

    number_of_links='number of links',

    number_of_hashtags='number of hashtags',

    emojis='emojis',

    pcfg_rules='pcfg rules',

    negativity_score='negativity score',

    neutrality_score='neutrality score',

    positivity_score='positivity score',

    part_of_speech_tagging='part of speech tagging'
)

TWEET_COLUMN_NAME = CSV_COLUMNS['tweet']
NUMBER_OF_WORDS_COLUMN_NAME = CSV_COLUMNS['number_of_words']
NUMBER_OF_LINES_COLUMN_NAME = CSV_COLUMNS['number_of_lines']
TWEET_HOUR_COLUMN_NAME = CSV_COLUMNS['tweet_hour']
TWEET_SOURCE_COLUMN_NAME = CSV_COLUMNS['tweet_source']
INCLUDE_MEDIA_COLUMN_NAME = CSV_COLUMNS['include_media']
NUMBER_OF_TAGS_COLUMN_NAME = CSV_COLUMNS['number_of_tags']
NUMBER_OF_LINKS_COLUMN_NAME = CSV_COLUMNS['number_of_links']
NUMBER_OF_HASHTAGS_COLUMN_NAME = CSV_COLUMNS['number_of_hashtags']
EMOJIS_COLUMN_NAME = CSV_COLUMNS['emojis']
NUMBER_OF_SHORT_WORDS_COLUMN_NAME = CSV_COLUMNS['number_of_short_words']
TWEET_LENGTH_COLUMN_NAME = CSV_COLUMNS['tweet_length']
PCFG_RULES_COLUMN_NAME = CSV_COLUMNS['pcfg_rules']
NEGATIVITY_SCORE_COLUMN_NAME = CSV_COLUMNS['negativity_score']
NEUTRALITY_SCORE_COLUMN_NAME = CSV_COLUMNS['neutrality_score']
POSITIVITY_SCORE_COLUMN_NAME = CSV_COLUMNS['positivity_score']
PART_OF_SPEECH_COLUMN_NAME = CSV_COLUMNS['part_of_speech_tagging']


USERS_FILE_PATH = Path(os.getcwd()) / 'data' / 'users.txt'
USERS_TWEETS_DIRECTORY_PATH = Path(
    os.getcwd()) / 'data' / 'output' / 'user-tweets'
ANALYSIS_FILE_PATH = Path(os.getcwd()) / 'data' / 'output' / "analysis.csv"
PARSER_DIRECTORY_PATH = Path(os.getcwd()) / "third_party" / \
    "stanford-parser-full-2018-10-17"
ARK_NLP_DIRECTORY_PATH = Path(os.getcwd()) / "third_party" / \
    "ark_tweet_nlp"
