#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/

Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""
import os
import subprocess
from pathlib import Path

import config

# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more
# useful (but not much).

# NOTE this command is directly lifted from runTagger.sh
RUN_TAGGER_CMD = 'java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2.jar --no-confidence  --output-format conll'
TMP_STRING_REPLACEMENT = 'lalalalala'


def _split_results(rows: list):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    accourances_counter = {}
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0 and not line.startswith(TMP_STRING_REPLACEMENT):
            if line.count('\t') == 1:
                parts = line.split('\t')
                tag = parts[1]
                if(tag in accourances_counter.keys()):
                    accourances_counter[tag] += 1
                else:
                    accourances_counter[tag] = 1
    return accourances_counter


def _call_runtagger(tweets: list, run_tagger_cmd: str = RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    tmp_file = open(config.TMP_POS_FILE, "w+")
    for tweet in tweets:
        if(tweet == ""):
            tmp_file.write(TMP_STRING_REPLACEMENT)
        else:
            tmp_file.write(tweet)
        tmp_file.write('\n')
    tmp_file.close()
    tmp_file_path = Path(os.getcwd()) / config.TMP_POS_FILE
    result = subprocess.check_output(run_tagger_cmd + ' ' + str(
        tmp_file_path), cwd=config.ARK_NLP_DIRECTORY_PATH, shell=True)
    os.remove(config.TMP_POS_FILE)
    # get first line, remove final double carriage return
    pos_results = result.decode('utf-8').strip('\n\n')
    split_to_tweets = pos_results.split('\r')
    split_to_words = [tweet.split('\n') for tweet in split_to_tweets]
    # remove last blank \n
    split_to_words = split_to_words[:-1]

    return split_to_words


def runtagger_parse(tweets: list, run_tagger_cmd: str = RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append(_split_results(pos_raw_result))
    return pos_result
