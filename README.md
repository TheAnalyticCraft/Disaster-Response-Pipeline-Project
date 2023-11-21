
# Disaster Response Pipeline Project

## Overview

This project is an integral part of the Data Science Nanodegree Program provided by Udacity in collaboration with Figure Eight. The dataset includes pre-labeled tweets and messages (~26K) collected from real disaster events. The main goal of this project is to create a Natural Language Processing (NLP) model capable of classifying messages and swiftly identifying post-disaster requirements in real-time scenarios.

**Project Components**:  

ETL Pipeline: *process_data.py*, serves as a data cleaning pipeline that:  
      - Loads the messages and categories datasets  
      - Merges the two datasets  
      - Transforms the data to generate category labels  
      - Cleans the data  
      - Stores it in a SQLite database  

ML Pipeline: *train_classifier.py*, serves as machine learning pipeline that:   
    - Loads the saved data from the SQLite database  
    - Splits the dataset into training and test sets  
    - Builds a text processing and machine learning pipeline  
    - Validate model results on the test set  
    - Exports the final model as a pickle file  
  
Flask Web App: *run.py*, executes the disaster response app :  
    - Generates data visualizations to provide insights into the data  
    - Allows users to input messages and retrieves predicted disaster categories

**Approach**:

In this project, a diverse set of classifiers, including RandomForest, GradientBoost, LinearSVC, and Logistic Regression, were utilized to determine the most suitable model for disaster message classification. The models were fine-tuned through a grid search involving specific parameters, with a focus on promising candidates identified during the initial assessment. Model selection was guided by performance metrics and considerations regarding computational resource requirements for training.

In disaster response scenarios characterized by imbalanced data, precision assumes a crucial role in guaranteeing predictions that optimize resource allocation. Equally pivotal is recall, as it ensures the identification of all genuine needs during critical situations. The F1 score, serving as a balanced metric that harmonizes precision and recall, proves especially pertinent when false positives and false negatives bear comparable consequences.

Simplifying the evaluation process, the focus was on assessing models using the F1 score for the positive label. This approach offered a straightforward gauge of each model's effectiveness. The evaluation prioritized models with a greater number of F1 scores surpassing the 0.5 threshold, denoting above-average performance and facilitating the selection of the most efficient models.


**Note**:

During exploring the data, it was found out that around 6,000 rows of messages didn't have labels.  These messages turned out to be not related to disasters, according to the source: https://github.com/rmunro/disaster_response_messages. So, only ~ 20K records that do have labels was utilized for the model training. 


## Getting Started

**Dependencies** 

- Python 3.6.3
- Main Python Packages:
  - Scikit-Learn
  - Flask
  - Plotly
  - Pandas
  - NumPy
  - NLTK
  - SQLalchemy

**Clone repository** 

```
git clone https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project.git
```

**Execution** 

- Run the following commands in the project's root directory to set up your database and model.  

    - To run ETL pipeline that cleans data and stores in database  
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories. csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves  
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

- Run the following command in the app's directory to run your web app. `python run.py`

- Go to http://0.0.0.0:3001/


**File Structure** 

app folder contains the following HTML `templates`:  
  - `index.html`: Renders homepage
  - `go.html`: Renders the message classifier  
  - `../app/run.py` is the script to execute the app  
data folder contains the following:  
  - `disaster_categories.csv`: disaster categories csv file
  - `disaster_messages.csv`: disaster messages csv file
  - `DisasterResponse.db`: database which is a merge of categories and messages 
  - `../data/process_data.py`: is the script for data cleaning
models folder contains the following:  
  - `classifier.pkl`: RandomForestClassifier pickle file
  - `train_classifier.py`: is script for model training


**Additional Materials**

- [ETL Pipeline Preparation.ipynb](https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/notebook/ETL%20Pipeline%20Preparation.ipynb): This notebook provides insights into the initial thought process behind the ETL (Extract, Transform, Load) part of the project, along with the experimentation of different procedures. The final output of this notebook is utilized to create the process_data.py file.  
- [ML Pipeline Preparation.ipynb](https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/notebook/ML%20Pipeline%20Preparation.ipynb): This notebook explains why specific machine learning algorithms were chosen and how they were incorporated into the model pipeline. It represents the initial thought process for building the model pipeline. The final outcome of this notebook serves as the foundation for creating the train_classifier.py file.

For a detailed explanation of the thought process and approach used, please refer to the markdown cells. 


## Screenshots:

<p align="center">
  <img src="" width="750" title="title">
</p>



## Model Results  

Note that the F1 scores for first 17 categories are above average, as shown below.

        feature                 precision   recall   f1_score   support   accuracy  
    27  earthquake              0.865     0.767    0.813      765       0.955  
    23  weather_related         0.845     0.692    0.761      2219      0.840  
    8   food                    0.694     0.810    0.748      878       0.920  
    1   aid_related             0.797     0.622    0.699      3284      0.708  
    25  storm                   0.702     0.684    0.693      751       0.925  
    7   water                   0.622     0.761    0.685      519       0.940  
    0   request                 0.703     0.666    0.684      1361      0.861  
    30  direct_report           0.623     0.615    0.619      1518      0.809  
    9   shelter                 0.547     0.683    0.608      761       0.889  
    24  floods                  0.538     0.658    0.592      637       0.904  
    14  death                   0.522     0.610    0.563      346       0.946  
    18  buildings               0.465     0.655    0.544      412       0.925  
    6   military                0.427     0.736    0.541      239       0.950  
    2   medical_help            0.475     0.604    0.532      594       0.895  
    28  cold                    0.487     0.583    0.531      156       0.973  
    10  clothing                0.425     0.658    0.517      117       0.976  
    19  electricity             0.454     0.575    0.507      179       0.967  
    3   medical_products        0.367     0.612    0.459      415       0.901  
    26  fire                    0.559     0.367    0.443      90        0.986  
    15  other_aid               0.386     0.483    0.430      1059      0.774  
    13  refugees                0.340     0.527    0.414      275       0.932  
    17  transport               0.328     0.452    0.380      354       0.913  
    29  other_weather           0.293     0.527    0.376      419       0.879  
    11  money                   0.307     0.456    0.367      180       0.953  
    16  infrastructure_related  0.264     0.456    0.334      566       0.830  
    20  hospitals               0.295     0.329    0.311      79        0.981  
    22  other_infrastructure    0.221     0.426    0.291      387       0.867  
    4   search_and_rescue       0.238     0.327    0.275      208       0.941  
    21  aid_centers             0.244     0.187    0.212      107       0.975  
    12  missing_people          0.185     0.207    0.195      82        0.977  
    5   security                0.131     0.216    0.163      134       0.951  


## Acknowledgements

* [Udacity](https://www.udacity.com/) for offering an exceptional Data Engineering Program
* [Figure Eight](https://www.figure-eight.com/) for providing the dataset essential for model training
