import csv
import random
from os import listdir
from os.path import isfile, join, splitext
 
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier

import config
from utils.features.training import training

_training = training()
existing_combinations = set()

tweet_files = [f for f in listdir(config.USERS_TWEETS_DIRECTORY_PATH) if isfile(
    join(config.USERS_TWEETS_DIRECTORY_PATH, f))]

with open('best_params.csv', mode='w') as params_analysis_file:
    csv_writer = csv.writer(
        params_analysis_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    i = 0
    while i < 5:
        # step 1 -> choose random people and import their tweet data
        analysis_row = []
        selected_tweet_files = random.sample(
            tweet_files, config.GROUP_SIZE)
        selected_tweet_files.sort()
        files_names_str = ", ".join(selected_tweet_files)

        if(files_names_str in existing_combinations):
            continue
        else:
            existing_combinations.add(files_names_str)

        all_tweets_data = list()

        Y = []
        for file_path in selected_tweet_files:
            y_user_name = splitext(file_path)[0]
            with open(join(config.USERS_TWEETS_DIRECTORY_PATH, file_path), newline='', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    all_tweets_data.append(row)
                    Y.append(y_user_name)
                csv_file.close()

        # step 2 -> vectorize all the data
        X = _training.vectorize_all_data(all_tweets_data, Y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=config.TEST_SIZE_RATIO, stratify=Y)

        # step 3 -> find best params
        parameters = {'hidden_layer_sizes': np.arange(80, 120, 10), 'alpha': np.arange(0, 2, 0.5), 'max_iter': [3000, 4000, 5000, 6000, 7000]}

        clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
        clf.fit(X_train, y_train)
        if i == 0:
            csv_writer.writerow([' ']+clf.cv_results_['params'])
        means = clf.cv_results_['mean_test_score']
        row = []
        row.append(files_names_str)
        for mean in means:
            row.append(mean)
        csv_writer.writerow(row)
        i += 1
