import os
from pathlib import Path

import config
from supervised import supervised
 
_supervised = supervised()
_group_sizes = [2, 5, 10, 20]

for size in _group_sizes:
    config.GROUP_SIZE = size

    config.USE_NLP_FEATURES = True
    config.USE_NGRAM_FEATURES = False
    config.USE_LEXICAL_FEATURES = False
    config.USE_TWITTER_FEATURES = False
    config.ANALYSIS_FILE_PATH = Path(
        os.getcwd()) / 'data' / 'output' / (str(size) + " only_nlp_features.csv")
    _supervised.run()

    config.USE_NLP_FEATURES = False
    config.USE_NGRAM_FEATURES = True
    config.USE_LEXICAL_FEATURES = False
    config.USE_TWITTER_FEATURES = False
    config.ANALYSIS_FILE_PATH = Path(
        os.getcwd()) / 'data' / 'output' / (str(size) + " only_ngram_features.csv")
    _supervised.run()

    config.USE_NLP_FEATURES = True
    config.USE_NGRAM_FEATURES = True
    config.USE_LEXICAL_FEATURES = True
    config.USE_TWITTER_FEATURES = False
    config.ANALYSIS_FILE_PATH = Path(
        os.getcwd()) / 'data' / 'output' / (str(size) + " nlp_ngram_lexical.csv")
    _supervised.run()

    config.USE_NLP_FEATURES = True
    config.USE_NGRAM_FEATURES = True
    config.USE_LEXICAL_FEATURES = True
    config.USE_TWITTER_FEATURES = True
    config.ANALYSIS_FILE_PATH = Path(
        os.getcwd()) / 'data' / 'output' / (str(size) + " all_features.csv")
    _supervised.run()
