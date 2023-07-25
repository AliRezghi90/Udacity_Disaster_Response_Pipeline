# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
nltk.download('stopwords') # download for stopwords 
from nltk.corpus import stopwords
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier

import pickle



def load_data(database_filepath):
    """ Function to load the tables
    Input: database filepath
    Output: dataframes of features X and target y
    """
    con = 'sqlite:///' + database_filepath
    engine = create_engine(con) # database
    df = pd.read_sql("SELECT * FROM DisasterResponseTable", engine) # dataframe
    X = df['message'] # X data
    # To get Y data, first find all te column names and select the last 36 columns
    colnames = df.columns.tolist() 
    Ycolnames = colnames[4:] 
    y = df[Ycolnames]
    return X, y, Ycolnames


def tokenize(text):
    """ 
    This function tokenizes the text data
    Inpute: text
    Output: a list of cleaned tokens (normalized, removed stopwords, lemmatized)
    """

    # Normalize text (remove punctuation characters and make lower case)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in words if word not in stopwords.words("english")]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer() #[WordNetLemmatizer().lemmatize(word) for word in tokens]
    
    clean_tokens = []
    for tok in tokens:
        ## lemmatize and remove leading/trailing white space
        # clean_tok = lemmatizer.lemmatize(tok).strip()  
        clean_tok = lemmatizer.lemmatize(tok, pos='v').strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model(pipeline_num=1):
    """ Function to build the classifier model
    Input: pipeline_num (1: RandomForestClassifier), (2: DecisionTreeClassifier)
    Output: pipeline
    """
    if (pipeline_num==1):
        pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
                        ])

        print('\n Pipelie parameters are: \n', pipeline.get_params()) # view the model parametes
    
    else:
        pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf_DT', MultiOutputClassifier(DecisionTreeClassifier()))
                    ])
        print('\nPipelie parameters are: \n', pipeline.get_params()) # view the model parametes
        
    return pipeline


def cal_score(y_test, y_pred):
    """ Function to calculate the scoring criteria for grid seaech
    Input: test and predicted columns
    Output: average of f1 score for all columns
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_test)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    return sum(f1_list)/len(f1_list)


def build_model_gridSearch(pipeline):
    """ Function to build the classifier model with grid search
    Input: nothing (should modify in the body)
    Output: gread search object
    """
    parameters = {
                'clf__estimator__class_weight': ['balanced'],
                'clf__estimator__min_samples_split': [2, 5, 8],
                'clf__estimator__n_estimators':[10, 50]
                }
    
    scoring = make_scorer(cal_score)
    cv =  GridSearchCV(pipeline, param_grid=parameters, verbose=1, scoring=scoring)
    cv.get_params().keys()

    return cv




def evaluate_model(model, X_test, y_test, category_names):
    """ Function to evaluate the model performance'
    Input: (fitted model: model), (test dataframe: X_test), (test dataframe: y_test),
            (A list of names of categories: category_names)
    Output: Averge accuracy of all categories, individual accuracies of all groups,
            Precision, recall, and f1-scofe of all labels in each category
    """
    
    y_pred = model.predict(X_test)
    
    accuracy_list = []; precision_list = []; recall_list = []; f1_list = []
    col_number = 0
    for col in category_names: 
        accuracy = (y_test[col]==y_pred[:,col_number]).mean()
        precision = precision_score(y_test[col], y_pred[:,col_number])
        recall = recall_score(y_test[col], y_pred[:,col_number])
        f1 = f1_score(y_test[col], y_pred[:,col_number])
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        col_number += 1
        
    scores = pd.DataFrame({'Category':category_names,
                          'Accuracy':accuracy_list,
                          'Precision': precision_list,
                          'Recall':recall_list,
                          'f1':f1_list})
 
    return scores


def display_results(model, scores):
    """ Function to display results
    Input: -
    Output: print statements for different scores
    """
    print('------------------- Results for the best model with X_test -------------------')
    print('Average Accuracy is:', scores['Accuracy'].mean())
    print('Average precision is:', scores['Precision'].mean())
    print('Average recall is:', scores['Recall'].mean())
    print('Average f1-score is:', scores['f1'].mean())
    print(scores)
    
    print('\n .... Best parameters for the model are:\n')
    for param in  model.best_params_.keys():
        print('\t' + param +': ', model.best_params_[param])
    
    print('\n ..... General model information:')
    print(model.cv_results_)

    
           
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)        




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        # database_filepath = 'sqlite:///DisasterResponseDatabase.db'
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        model_name = 'RandomForest'
        gridsearch = 1

        if gridsearch==1:
            print('\n Building model', model_name, ' with grid search ....\n')
            model= build_model(pipeline_num=1)
            cv = build_model_gridSearch(model)
            
            print('\n Training model', model_name, ' with grid search ....\n')
            cv.fit(X_train, y_train)
            
            print('\n Evaluating model', model_name, 'with grid search.... \n')
            scores = evaluate_model(cv, X_test, y_test, category_names)
            


            print('\n Saving model', model_name, '....\n')
            save_model(cv, model_filepath)
            
            display_results(cv, scores)
            print('\n\n .... Trained model saved!')
        
        else: # Only pipeline (not grid search)  
            print('\n Building model', model_name, '....\n')
            model= build_model(pipeline_num=1)
            print('\n Training model', model_name, '....\n')
            model.fit(X_train, y_train)
            print('\n Evaluating model', model_name, '.... \n')
            scores = evaluate_model(model, X_test, y_test, category_names)
            print(scores)
            print('\n Saving model', model_name, '....\n')
            save_model(model, model_filepath)
            print('\n\n .... Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()


# Run this command to run the code (navigate to the folder first):
# python .\models\train_classifier.py .\data\DisasterResponse.db .\models\classifier.pkl