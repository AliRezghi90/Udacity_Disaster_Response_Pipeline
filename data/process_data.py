import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Function to load messages and categorie data sets
    Input: messages and categorie csv files
    Output:  merged dataframe from messages and categorie datasets
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id') # merge datasets
    # messages.shape
    # messages.isnull().sum()
    # categories.shape
    # categories.isnull().sum()
    print('\n... Dataframe created successfully ...\n')

    return df


def change_url(text):
    """ Function to convert url linke (https:...) to a a string (urlplaceholder)
    Input: text string
    Output: text url link changed to "urlplaceholder"
    """
    
    # regular expression to detect a url
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_regex2 = 'http.*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    detected_urls2 = re.findall(url_regex2, text)
    detected_urls.extend(detected_urls2)
    

    
    # replace each url in text string with placeholder
    if detected_urls != []:
        for url in detected_urls:
            text = text.replace(url, "urlplaceholder")
            return text
    else:
        return text
    
    

def clean_data(df):
    """ Function to clean dataframe 
    Input: df dataframe
    Output: Clean dataframe df with seperated categories, 
            numeric column values, and removed duplicates
    """

    ## 1. Split `categories` into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = df.loc[0, 'categories']
    # use this row to extract a list of new column names for categories.
    category_colnames = re.sub("[-, \d]",'', row).split(';')
    # rename the columns of `categories`
    categories.columns = category_colnames
    # print('... Splited `categories` into separate category columns ...')

    ## 2. Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # print('... Converted category values to just numbers 0 or 1 ...')
    
    ## 3. Replace `categories` column in `df` with new category columns:
            # - Drop the categories column from the df dataframe since it is no longer needed.
            # - Concatenate df and categories data frames.
    df.drop(['categories'], axis = 1, inplace=True)
    df = pd.concat([df, categories], axis = 1)

    ## 4. Remove duplicates.
        # - Check how many duplicates are in this dataset.
        # - Drop the duplicates.
        # - Confirm duplicates were removed.
    # print('......Number of duplicates in the dataframe before removing duplicates is: ', df.duplicated().sum(), '\n')
    df.drop_duplicates(inplace=True)
    # print('......Number of duplicates in the dataframe after removing duplicates is: ', df.duplicated().sum(),'\n')
    # print('... Removed duplicate values ...')

    # all the values of category "child_alone" are zero
    df.drop(['child_alone'], axis = 1, inplace=True, errors='ignore')
    
    # The "related" category has also 2 values. Let's change them to 1 as it is the majority.
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)


    ## change url links (http.....) with string "urlplaceholder"
    ## iterated 10 time only because some message have several spaces between
    ## "http" and the rest of the url link (check later for a better solution)
    for i in range(1,10):
        df['message'] = df['message'].map(change_url)

    return df



def save_data(df, database_filename):
    """ Function to save the clean dataset into an sqlite database
    Input: cleaned dataframe df
    Output: DisasterResponse Database and DisasterResponse Table
    """

    con = 'sqlite:///'+ database_filename
    engine = create_engine(con)
    return df.to_sql('DisasterResponseTable', engine, index=False,  if_exists='replace')




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]


        print('\n Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('\n Cleaning data...\n')
        df = clean_data(df)
        
        print('\n Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('\nCleaned data and saved to database!....\n')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()



# Run this command to run the code (navigate to the folder first):
# python .\data\process_data.py .\data\disaster_messages.csv .\data\disaster_categories.csv .\data\DisasterResponse.db
