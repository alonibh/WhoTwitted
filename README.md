# The Who Tweeted Porject

In this project we attempted to find the best method for user classification in Twitter. \
We used the Twitter API that helped us extracting the latest 3,200 tweets per user. \
Full documentation can be found here : https://developer.twitter.com/en/docs/tweets/timelines/api-reference/get-statuses-user_timeline 

## prerequisite
Python 3.7\
Java\
Pip packages:
* sklearn
* nltk
* tweepy
* vaderSentiment
* tweet-preprocessor (from '\third_party\tweet-preprocessor-0.5.0\' - pip install .)


This project contains 2 main uses: creating the dataset and running the supervised machine learning with it. 
## How to create the dataset
Simply run create_data_set.py, this script uses 'users.txt' located in 'data/' directory \
users.txt contains list of user names, each line contains different Twitter user name. \
The output for this script in a single CSV file for each user. \
Each CSV contains all of the data needed for the classifiers to create the features vector. \
Every line in the CSV file represents different tweet. \
The script output is written by default to 'data/output/user-tweets' directory. 


## How to run the supervised maching learning algorithm
Run the run_supervised.py script. This script reads the dataset that was created and runs the supervised machine learning methods and fits the model with it. \
The algorithm uses 3 different Machine Learning methods (MLPClassifier, LogisticRegression and ExtraTreesClassifier) by default and can be extended to other classifiers. \
Eventually the script creates multiple CSV file for each of the predefined group sizes, that contains the average accuracy score for each classifier. 

Properties such as: 
* number of groups
* classifiers
* output directory
* number of tweets extracted per user
* test size ratio 

can be modified in the config.py file

