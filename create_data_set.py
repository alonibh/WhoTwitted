import config 
from utils.data_set.data_set_builder import create_dataset

create_dataset(config.USERS_FILE_PATH, config.USERS_TWEETS_DIRECTORY_PATH)
print("Done!") 
