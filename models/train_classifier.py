
# import libraries

import sys
import re
import pickle
import os

from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def load_data(database_filepath):

    """
    Load data from a SQLite database and return feature and target variables.

    Parameters:
    database_filepath (str): File path of the SQLite database

    Returns:
    X (pd.Series): Feature variable (messages)
    y (pd.DataFrame): Target variables (categories)
    category_names (list): List of category names
    """

    try:

        # Create an SQLAlchemy engine
        engine = create_engine(f'sqlite:///{database_filepath}')

        # Extract the table name from the database filepath (excluding extension)
        table_name = os.path.splitext(os.path.basename(database_filepath))[0]

        # Save the DataFrame to the database, replace if it already exists
        df = pd.read_sql(f'select * from {table_name}', engine)    

        # Identify rows with missing labels
        df['no_label'] = df.iloc[:,4:].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

        # Remove those rows
        df = df[df['no_label'] != 1].reset_index()

        # Drop Child Alone and Related columns
        df = df.drop(['child_alone','related','no_label'],axis=1)

        X = df['message']
        y = df.iloc[:,5:]
        category_names = y.columns

        return X, y , category_names
    
    except Exception as e:
        print(f"Error loading data from database: {e}")
        sys.exit(1)

def tokenize(text):

    """
    This function performs custom text tokenization by normalizing text, removing specified patterns,
    removing stopwords, tokenizing, and lemmatizing the text.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list of str: List of tokenized and lemmatized words.
    """    

    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    phone_pattern =  r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    email_pattern = r'[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+'
    
    # normalize text
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    text = re.sub(r'(?:\b\d+\b)', ' ', text)    
    #text = re.sub(r'\s\d+(\s\d+)*\s', ' ', text)
    
    for regexp in [url_pattern, phone_pattern, email_pattern]:            
        patterns = re.findall(regexp, text)
        for extract in patterns:
            text = text.replace(extract, ' ')
            
    # stopword list 
    stop_words = stopwords.words("english")
        
    # tokenize
    words = word_tokenize(text)
        
    # lemmatize
    words_lemmed = [WordNetLemmatizer().lemmatize(w).strip() for w in words if w not in stop_words]

    return words_lemmed


def build_model():

    """
    Build a machine learning pipeline with TfidfVectorizer and MultiOutputClassifier.

    Returns:
    pipeline: Machine learning pipeline
    """

    classifier = RandomForestClassifier()
    grid_params = {
        'estimator__estimator__max_depth': [5],            
        'estimator__estimator__n_estimators': [500],
        'estimator__estimator__min_samples_split': [25],
        'estimator__estimator__class_weight': [None, 'balanced'],
        'vectorizer__max_features': [1000,3000],
        'vectorizer__ngram_range': [(1, 2)],
        'vectorizer__min_df': [6]
    }    

    # Standard Pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize)),
        ('estimator', MultiOutputClassifier(classifier))
    ])

    # Apply GridSearchCV if needed
    grid_cv = GridSearchCV(pipeline, param_grid=grid_params, cv=3)  # Adjust cv as needed

    return grid_cv


def evaluate_model(model, X_test, y_test):

    """
    Evaluate the model's performance on the test set and print out the metrics.

    Parameters:
    model: Trained machine learning model
    X_test: Test set features
    y_test: Test set labels
    """    

    y_pred = model.predict(X_test)

    print('Model Performance:','\n')
    
    metrics_ = []
    for i in range(y_test.shape[1]):
        precision_, recall_, f1_score_, support_ = precision_recall_fscore_support(y_test.iloc[:,i], y_pred[:,i])        
        acc_ = accuracy_score(y_test.iloc[:, i], y_pred[:, i])            
        metrics_.append([y_test.columns[i], precision_[1], recall_[1], f1_score_[1], support_[1], acc_])  

    # read into pandas DF
    model_metrics = pd.DataFrame(metrics_, columns=['feature','precision', 'recall', 'f1_score','support', 'accuracy'])

    # 'support' should be an integer
    model_metrics['support'] = model_metrics['support'].astype(int)
    model_metrics.sort_values(by=['f1_score'], ascending=False, inplace=True)

    f1_count = sum(np.where(model_metrics['f1_score'] >= 0.5, 1,0))
    print(f"No. of Catgeories (n = {model_metrics.shape[0]}) with F1 scores Above Average {f1_count}", '\n')

    print(model_metrics,'\n')

    return 


def save_model(model, model_filepath):

    """
    Save the trained model as a pickle file.

    Parameters:
    model: Trained machine learning model
    model_filepath (str): File path to save the pickle file
    """    

    try:
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        return
    except Exception as e:
        print(f"Error saving model to {model_filepath}: {e}")
        sys.exit(1)


def main():

    " Approximatly 16 minutes to run GridSearch"

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data... DATABASE: {}'.format(database_filepath))

        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)        

        print('Building model...')
        pipeline = build_model()
        
        print('Training model...')
        model = pipeline.fit(X_train, y_train)
        
        print('Evaluating model...','\n')
        evaluate_model(model, X_test, y_test)

        print('Saving model... MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('*** Trained Model Saved ***')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()