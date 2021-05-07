import csv
import operator as op
import random
from datetime import datetime
from functools import reduce
from os import listdir
from os.path import isfile, join, splitext

import sklearn.metrics
from sklearn.model_selection import train_test_split

import config
from utils.features.training import training


class supervised():
    _training = training()

    def _ncr(self, n, r):
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

    def _train_and_predict(self, X: list, Y: list):
        accuracy_score_dict = dict()
        print('-----------------')
        for classifier in config.CLASSIFIERS:
            classifier_name = str(classifier).split("(")[0]
            accuracy_score_sum = 0
            for i in range(config.NUMBER_OF_ITERATIONS_FOR_PREDICTING):
                print(classifier_name + " - Iteration #" + str(i+1))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, Y, test_size=config.TEST_SIZE_RATIO, stratify=Y)

                # step 5 -> train the data
                model = self._training.train(X_train, y_train, classifier)

                # step 6 -> predict the y by the x_test
                y_pred = model.predict(X_test)

                # step 7 -> find accuracy score
                accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
                print("Accuracy Score : " + str(accuracy_score))
                accuracy_score_sum += accuracy_score
            print("Classifier's average accuracy scores " +
                  str(accuracy_score_sum/config.NUMBER_OF_ITERATIONS_FOR_PREDICTING))
            print('-----------------')
            accuracy_score_dict[str(classifier)] = accuracy_score_sum / \
                config.NUMBER_OF_ITERATIONS_FOR_PREDICTING
        return accuracy_score_dict

    def run(self):
        # step 1 -> create the dataset
        # python .\create_data_set.py
        existing_combinations = set()

        tweet_files = [f for f in listdir(config.USERS_TWEETS_DIRECTORY_PATH) if isfile(
            join(config.USERS_TWEETS_DIRECTORY_PATH, f))]
        max_number_of_groups = self._ncr(len(tweet_files), config.GROUP_SIZE)

        if config.NUMBER_OF_GROUPS > max_number_of_groups:
            print("Configuration error, attempt to create " + str(config.NUMBER_OF_GROUPS) +
                  " groups while there are only " + str(max_number_of_groups) + " combinations")
            exit(0)

        with open(config.ANALYSIS_FILE_PATH, mode='w+', newline='', encoding='utf-8') as analysis_file:
            analysis_writer = csv.writer(
                analysis_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            headers = ["group names"]

            for c in config.CLASSIFIERS:
                headers.append(str(c).split("(")[0])
            analysis_writer.writerow(headers)

            i = 0
            while i < config.NUMBER_OF_GROUPS:
                # step 2 -> choose random people and import their tweet data
                analysis_row = []
                selected_tweet_files = random.sample(
                    tweet_files, config.GROUP_SIZE)
                selected_tweet_files.sort()
                files_names_str = ", ".join(selected_tweet_files)

                if(files_names_str in existing_combinations):
                    continue
                else:
                    existing_combinations.add(files_names_str)

                print("*******************")
                print("starting group #" + str(i+1))
                files_names_replace = files_names_str.replace(".csv", "")
                print("The group : " + files_names_replace)
                startIter = datetime.now()
                print(startIter)
                all_tweets_data = list()

                Y = []
                for file_path in selected_tweet_files:
                    y_user_name = splitext(file_path)[0]
                    with open(join(config.USERS_TWEETS_DIRECTORY_PATH, file_path), newline='', encoding='utf-8') as csv_file:
                        reader = csv.DictReader(csv_file)
                        num_of_rows=0
                        for row in reader:
                            if num_of_rows<config.VECTOR_SIZE:
                                all_tweets_data.append(row)
                                Y.append(y_user_name)
                                num_of_rows+=1
                            else:
                                break
                        csv_file.close()
                
                # step 3 -> vectorize all the data

                X = self._training.vectorize_all_data(all_tweets_data, Y)

                # step 4 -> find accuracy score
                current_accuracy_scores = self._train_and_predict(X, Y)
                analysis_row.append(files_names_replace)

                for _, accuracy_score in current_accuracy_scores.items():
                    analysis_row.append(accuracy_score)

                analysis_writer.writerow(analysis_row)
                print("finished group #" + str(i + 1) +
                      " in " + str(datetime.now()-startIter))

                i += 1

        print("DONE")
