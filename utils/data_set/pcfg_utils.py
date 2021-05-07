import os
import subprocess
from heapq import nlargest

import config
from utils.tree.builders import list_tree_from_sequence
from pathlib import Path

# create txt file of the rules counter of a specific tweeter


def extract_pcfg_trees_from_tweet(tweet: str):
    # Create a temp file in order for the process to read it
    tmp_file = open(config.TMP_TWEET_FILE, "w+")
    tmp_file.write(tweet)
    tmp_file.close()
    tmp_file_path = Path(os.getcwd()) / config.TMP_TWEET_FILE

    tt = subprocess.Popen('java -mx150m -cp "*;" edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ' + str(tmp_file_path),
                          shell=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          stdin=subprocess.PIPE,
                          cwd=config.PARSER_DIRECTORY_PATH)
    output, _ = tt.communicate()
    output = str(output)
    output = output.replace('\\n', ' ').replace(
        '\\r', '').replace("b'", '').replace("'", "")
    os.remove(config.TMP_TWEET_FILE)
    pcfg_trees = list()
    new_tree_indexes = get_new_tree_indexes(output)
    for i in range(len(new_tree_indexes)):
        tree_to_add = str()
        if i == len(new_tree_indexes) - 1:
            tree_to_add = output[new_tree_indexes[i]:]
        else:
            tree_to_add = output[new_tree_indexes[i]:new_tree_indexes[i+1]]
        tree_to_add = tree_to_add.replace("( )", "").replace("()", "")
        pcfg_trees.append(tree_to_add)
    return pcfg_trees

# recieves rules dictionary and returnes array of the rules
def convert_top_rules_to_vector(rules_dict):
    rules_vector = []
    for a_rule in rules_dict.keys():
        for b_rule in rules_dict[a_rule]:
            rule_str = a_rule + " " + b_rule
            rules_vector.append(rule_str)
    return rules_vector

# recieves dictionary and vectorizes it by the list of top rules
def get_rules_vectorized(rules_dict: dict, all_top_rules: list):
    rules_vector = []
    for top_rule in all_top_rules:
        a_rule = top_rule.split()[0]
        b_rule = top_rule.split()[1]
        if a_rule in rules_dict.keys():
            if b_rule in rules_dict[a_rule].keys():
                rules_vector.append(rules_dict[a_rule][b_rule])
                continue
        rules_vector.append(0)
    return rules_vector

# return all of the indexes of the begining of new trees (ROOT)
def get_new_tree_indexes(trees_str: str):
    indexes = list()
    for i in range(len(trees_str)):
        if trees_str[i:].startswith('(ROOT'):
            indexes.append(i)
    return indexes

# combine the two dictionaries and sums their occurances
def combine_rules(first_rules_dict: dict, second_rules_dict: dict):
    combined_rules_dict = dict()
    for base, derived in first_rules_dict.items():
        combined_rules_dict[base] = derived
    for base, derived in second_rules_dict.items():
        for key, occurrences in derived.items():
            if combined_rules_dict.get(base) is not None:
                if combined_rules_dict[base].get(key) is not None:
                    combined_rules_dict[base][key] += occurrences
                else:
                    combined_rules_dict[base][key] = occurrences
            else:
                combined_rules_dict[base] = dict()
                combined_rules_dict[base][key] = occurrences
    return combined_rules_dict


# receive list of PCFG trees as strings and return rules counter for all of them (except for rules that are being excluded - according to config)
def extract_rules_from_pcfg_trees(pcfg_trees: list):
    rules_dict = {}
    for pcfg_tree in pcfg_trees:
        sentence_tree = list_tree_from_sequence(pcfg_tree)
        extract_rules_from_sentence_tree(
            sentence_tree, rules_dict)

    only_valid_rules_dict = {}
    for base, keys in rules_dict.items():
        for derived, occurrences in keys.items():
            if '*' in derived:
                continue
            if base in config.FREQUANT_PCFG_RULES.keys():
                if derived in config.FREQUANT_PCFG_RULES[base]:
                    continue
            if only_valid_rules_dict.get(base) is None:
                only_valid_rules_dict[base] = dict()
            only_valid_rules_dict[base][derived] = occurrences

    return only_valid_rules_dict

# receive a sentence tree (ROOT (PRP I)) and return its rules
def extract_rules_from_sentence_tree(sentence_tree: list, rules_dict: dict):
    if(type(sentence_tree) is str):
        return
    if(not sentence_tree.children):
        return
    if(len(sentence_tree.children) == 0):
        return
    children_str = ""
    first_flag = True
    for child in sentence_tree.children:
        if(first_flag):
            first_flag = False
        else:
            children_str += "#"
        if(type(child) is str):
            children_str += "*" + child
        else:
            children_str += child.head
        extract_rules_from_sentence_tree(child, rules_dict)
    if(not sentence_tree.head in rules_dict):
        rules_dict[sentence_tree.head] = {}
        rules_dict[sentence_tree.head][children_str] = 1
    elif(not children_str in rules_dict[sentence_tree.head]):
        rules_dict[sentence_tree.head][children_str] = 1
    else:
        rules_dict[sentence_tree.head][children_str] += 1


def get_most_frequent_pcfg_rules(pcfg_rules: dict, N: int):
    all_rules = dict()
    for base, derived in pcfg_rules.items():
        for key, occurrences in derived.items():
            all_rules[(base, key)] = occurrences
    n_most_frequent = nlargest(N, all_rules, key=all_rules.get)
    return n_most_frequent
