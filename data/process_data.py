
# import libraries

import sys
import os

import pandas as pd

from sqlalchemy import create_engine


# Function to load files
def load_data(msg_filepath, categories_filepath):
    '''
    Load and merge data from two CSV files.

    Parameters:
        msg_filepath (str): File path to the messages CSV file.
        categories_filepath (str): File path to the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing the data from both files.
    '''

    try:
        # Load messages dataset
        messages = pd.read_csv(msg_filepath)
        print('Loaded messages file:', msg_filepath)

        # Load categories dataset
        categories = pd.read_csv(categories_filepath)
        print('Loaded categories file:', categories_filepath)

        # Merge datasets on the 'id' column
        df = messages.merge(categories, on='id')

        return df
    
    except Exception as e:
        print('An error occurred while reading or merging files:', str(e) )
        sys.exit(1)


def clean_data(df):
    '''
    Applies data processing to clean the dataset for model training.

    Parameters:
        df (pd.DataFrame): Output of the load_data function.

    Returns:
        pd.DataFrame: Ready dataset for model training.
    '''

    try:
        # Check if the categories column is in the DataFrame            
            
        if 'categories' in df.columns:
            # Split the categories column into separate columns
            df_split = df['categories'].str.split(';', expand=True)

            # Extract the column names from the first row
            col_names = df_split.loc[0].str.split('-', expand=True)[0].tolist()

            # Convert category columns to numeric and replace values greater than 1 with 1
            for col in df_split.columns:
                df_split[col] = df_split[col].apply(lambda x: int(x[-1]))
                df_split[col] = df_split[col].apply(lambda x: 1 if x > 1 else x)

            # Assign column names to the split DataFrame
            df_split.columns = col_names

            # Drop the original 'categories' column
            df_clean = df.drop('categories', axis=1)

            # Merge the split categories with the original DataFrame
            df_clean = pd.concat([df_clean, df_split], axis=1)

            # Remove duplicate rows
            df_clean = df_clean.drop_duplicates()
            
            return df_clean

    except Exception as e:
        print('An error occurred in data cleaning:', str(e))
        sys.exit(1)


def save_data(df, database_filepath):
    '''
    Save a Pandas DataFrame to a SQLite database.

    Parameters:
        df (pd.DataFrame): The DataFrame to be saved.
        database_filepath (str): The path to the SQLite database file.

    Returns:
        None
    '''

    try:
        # Check if DataFrame is not None and has data
        if df is not None and not df.empty:

            # Create an SQLAlchemy engine
            engine = create_engine(f'sqlite:///{database_filepath}')

            # Extract the table name from the database filepath (excluding extension)
            table_name = os.path.splitext(os.path.basename(database_filepath))[0]

            # Save the DataFrame to the database, replace if it already exists
            df.to_sql(table_name, engine, index=False, if_exists='replace')
            print('Dataset Saved:', table_name)

            return None

    except Exception as e:
        print('An error occurred in data saving: ', str(e))
        sys.exit(1)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Message : {}\n    Categories : {}'
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')
        print('Invalid number of arguments. Script terminated.')

if __name__ == '__main__':
    main()
