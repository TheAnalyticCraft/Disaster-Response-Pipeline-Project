# import libraries

import re
import pickle
import os
import sys
import json

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request

import plotly
from plotly.graph_objs import Bar, Layout
from sklearn.externals import joblib
from sqlalchemy import create_engine

import pandas as pd


app = Flask(__name__)

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


# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# --------------------------------------------------------------------------------------------------------------------#
# Identify rows with missing labels
df['no_label'] = df.iloc[:,4:].apply(lambda x: 1 if x.sum() == 0 else 0, axis=1)

# Remove those rows
df = df[df['no_label'] != 1]

# Remove non-essential columns
df_drop = df.drop(['child_alone', 'related', 'no_label'],axis=1)
df = df.drop(['related', 'no_label'],axis=1)
df['n_labels'] = df.iloc[:,4:].sum(axis=1)


# --------------------------------------------------------------------------------------------------------------------#

# Summarize Data - First Chart (% Classified Messaged By Category)

category_values = df.iloc[:,4:]
category_mean = category_values.mean().sort_values(ascending=False).reset_index()
category_mean.columns = ['category', 'mean_response']

# Create a mapping dictionary
mapping = {
    'request': ['request', 'related'],
    'offer': ['offer'],
    'aid': [
        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 
        'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 
        'missing_people', 'refugees', 'death', 'other_aid'
    ],
    'infrastructure': [
        'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals',
        'shops', 'aid_centers', 'other_infrastructure'
    ],
    'weather': ['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather'],
    'report': ['direct_report']
}

# Create a reverse mapping dictionary
reverse_mapping = {value: key for key, values in mapping.items() for value in values}

# Map the first column (category) to major groups using reverse_mapping
category_mean['category_class'] = category_mean['category'].map(reverse_mapping)
category_mean = category_mean[category_mean['category'] != 'related']
category_mean.sort_values(by=['category_class', 'mean_response'], ascending=[True, False], inplace=True)

# --------------------------------------------------------------------------------------------------------------------#

# Summarize Data - Second Chart (Number of Classifications)

nmsg_groups  = df['n_labels'].sort_values().value_counts().reset_index()
nmsg_groups.columns = ['group', 'nmsg']
nmsg_groups.sort_values(by='nmsg', ascending=False, inplace=True)

# --------------------------------------------------------------------------------------------------------------------#

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    """
    Index webpage that displays cool visuals and receives user input text for the model.
    
    Returns:
        render_template: Flask template to render the webpage with plotly graphs.
    """

    # Define a color map for the category classes
    bar_color = {
        'request': '#ADD8E6',
        'offer': '#000000',
        'aid': '#D2681E',
        'infrastructure': '#C20078',
        'weather': '#DBB40C',
        'report': '#030764'
    }

    # create the first chart

    fst_data = []

    for category_class, group in category_mean.groupby('category_class'):
        # Use the color map to get the color for this class
        class_color = bar_color[category_class]

        fst_data.append(Bar(
            x=group['category'],
            y=group['mean_response'],
            name=category_class,  # This will be used in the legend
            marker=dict(color=class_color)  # Set the color for this group
        ))

    fst_layout = Layout(
        title='Messages Classified By Categories',
        #width=800,
        #height=500,
        margin=dict(l=50, r=40, t=40, b=120),
        yaxis=dict(
            title='% Classified',
            range=[0, .75],
            tickformat='.0%',
            titlefont=dict(size=12),
            tickfont=dict(size=9, color='black')
        ),
        xaxis=dict(
            tickangle=90,
            tickfont=dict(size=9, color='black')
        ),
        # Add a horizontal line at y=0.05
        shapes=[
            dict(
                type='line',
                yref='y', y0=0.05, y1=0.05,
                xref='paper', x0=0, x1=1,
                line=dict(
                    color='Black',
                    width=1,
                    dash='dashdot',
                )
            )
        ],
        legend=dict(
            orientation='h',
            font=dict(size=11, color='Black'),
            x=0.2,  # Fractional x position
            y=.85,    # Fractional y position
            bgcolor='rgba(255, 255, 255, 0.5)',  # Optional: semi-transparent background
            #bordercolor='',  # Optional: border color
            borderwidth=0  # Optional: border width
        ),
        # Add an annotation for the horizontal line
        annotations=[
            dict(
                xref='paper', x=0.4,  # Position the x based on the figure's width
                yref='y', y=0.06,  # Position the y at the horizontal line's level
                text='Severe Cases of Unbalanced Data',  # The text of the annotation
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30  # Adjust the arrow's position
            )
        ]    
    )

    fst_chart = {'data': fst_data, 'layout': fst_layout}

    # Create second chart

    sec_data = [
        Bar(
            x=nmsg_groups.group,
            y=nmsg_groups.nmsg,
            marker=dict(color='#06C2AC')
        )     
    ]

    sec_layout = Layout(
        title='Analysis of Messages Based on the Number of Classified Categories',
        titlefont= dict(size=13), 
        barmode='stack',
        margin= dict(l=75, r=20, t=30, b=40),   
        #width= 800,
        #height= 400,
        showlegend=False,
        yaxis = dict(range=[0, nmsg_groups['nmsg'].max()+1000]),
        xaxis=dict(
            title='# of Categories a Message is Classified',        
            titlefont=dict(size=13),
            tickfont=dict(size=10, color='black'),
            range=[0,34]
        )  
        #legend= dict(font=dict(size=11))         
    )

    sec_chart = {'data': sec_data, 'layout': sec_layout}

    graphs = [fst_chart, sec_chart]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query    
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df_drop.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='0.0.0.0', port=3000, debug=True)
    app.run(debug=True)


if __name__ == '__main__':
    main()