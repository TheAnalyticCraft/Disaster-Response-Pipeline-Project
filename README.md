
# Disaster Response Pipeline Project 

## Overview

This project is an integral part of the Data Engineering course provided by Udacity in collaboration with Figure Eight. The dataset includes pre-labeled tweets and messages (~26K) collected from real disaster events. The main goal of this project is to create a Natural Language Processing (NLP) model capable of classifying messages and swiftly identifying post-disaster requirements in real-time scenarios.

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
    - Trains various classifiers and fine-tunes them using GridSearchCV  
    - Validates model results on the test set  
    - Exports the final model as a pickle file  
  
Flask Web App: *run.py*, executes the disaster response app :  
    - Generates data visualizations to provide insights into the data  
    - Allows users to input messages and retrieves predicted disaster categories

**Approach**:

In this project, a diverse set of classifiers, including RandomForest, GradientBoost, LinearSVC, and Logistic Regression, were utilized to determine the most suitable model for disaster message classification. The models were fine-tuned through a grid search involving specific parameters, with a focus on promising candidates identified during the initial assessment. Model selection was guided by performance metrics and considerations regarding computational resource requirements for training.

The disaster dataset exhibits a significant class imbalance, with some categories having a very small occurence rate (< 10% or < 5%). When working with imbalanced data, it is crucial to use appropriate metrics for model evaluation. Precision is essential for ensuring accurate predictions, optimizing resource allocation in disaster response scenarios. Recall is equally important as it ensures the identification of all genuine needs during critical situations. The F1 score, a balanced metric that combines precision and recall, becomes particularly relevant when false positives and false negatives have similar consequences.

Simplifying the evaluation process, the focus was on assessing models using the F1 score for the minority label. This approach offered a straightforward gauge of each model's effectiveness. The evaluation prioritized models with a greater number of F1 scores surpassing the 0.5 threshold, denoting above-average performance and facilitating the selection of the most efficient models.  

To mitigate the severity of class imbalance, a parameter called "class weight" was specified, resulting in improved model performance, especially in the case of Random Forest.


**Note**:

During the data exploration phase, it became evident that approximately 6,000 rows of messages were unrelated to disasters, as confirmed by the source at https://github.com/rmunro/disaster_response_messages. Additionally, approximately 5,000 messages directly related to disasters lacked category labels. Consequently, only 20,000 disaster-related data points were included for model training. It's noteworthy that during this process, two columns, 'child_alone' and 'related,' were excluded from the dataset.


**Jupyter Notebooks**

- [ETL Pipeline Preparation.ipynb](https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/notebook/ETL%20Pipeline%20Preparation.ipynb): This notebook provides insights into the initial thought process behind the ETL (Extract, Transform, Load) part of the project, along with the experimentation of different procedures. The final output of this notebook is utilized to create the process_data.py file.  
- [ML Pipeline Preparation.ipynb](https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/notebook/ML%20Pipeline%20Preparation.ipynb): This notebook explains why specific machine learning algorithms were chosen and how they were incorporated into the model pipeline. It represents the initial thought process for building the model pipeline. The final outcome of this notebook serves as the foundation for creating the train_classifier.py file.

For a detailed explanation of the thought process and approach used, please refer to the markdown cells. 

## Model Results  

The final model, selected through grid search on a Random Forest classifier, achieved above-average F1 scores for the top 16 out of 34 categories, indicating its effectiveness in classifying disaster-related messages. These results highlight the model's strong performance in identifying critical information within the dataset.  

In addition to assessing model performance, it's crucial to engage in discussions with disaster response teams to present the initial findings, including the model's strengths and limitations. These conversations should also focus on exploring the practical applications of the predictions in real-time scenarios and determining the best approach for operationalizing these predictions, with input and collaboration from the teams.  


| index\category         | precision | recall | f1_score | support | accuracy |
|------------------------|-----------|--------|----------|---------|----------|
| 30 | earthquake        | 0.859     | 0.77   | 0.812    | 722     | 0.957    |
| 26 |weather_related    | 0.857     | 0.703  | 0.772    | 2149    | 0.852    |
| 9  |food               | 0.743     | 0.797  | 0.769    | 896     | 0.929    |
| 8  |water              | 0.66      | 0.786  | 0.717    | 504     | 0.948    |
| 28 |storm              | 0.702     | 0.73   | 0.715    | 732     | 0.93     |
| 2  |aid_related        | 0.81      | 0.612  | 0.697    | 3237    | 0.714    |
| 0  |request            | 0.694     | 0.667  | 0.681    | 1347    | 0.86     |
| 10 |shelter            | 0.556     | 0.737  | 0.634    | 712     | 0.899    |
| 33 |direct_report      | 0.62      | 0.596  | 0.608    | 1549    | 0.802    |
| 27 |floods             | 0.495     | 0.702  | 0.581    | 625     | 0.895    |
| 11 |clothing           | 0.5       | 0.691  | 0.58     | 136     | 0.977    |
| 15 |death              | 0.515     | 0.613  | 0.56     | 388     | 0.938    |
| 7  |military           | 0.418     | 0.742  | 0.534    | 236     | 0.949    |
| 3  |medical_help       | 0.488     | 0.571  | 0.527    | 630     | 0.893    |
| 31 |cold               | 0.429     | 0.682  | 0.526    | 154     | 0.969    |
| 19 |buildings          | 0.425     | 0.676  | 0.521    | 404     | 0.917    |
| 4  |medical_products   | 0.354     | 0.652  | 0.459    | 397     | 0.899    |
| 29 |fire               | 0.481     | 0.432  | 0.455    | 88      | 0.985    |
| 16 |other_aid          | 0.366     | 0.526  | 0.432    | 1001    | 0.77     |
| 12 |money              | 0.316     | 0.59   | 0.411    | 161     | 0.955    |
| 20 |electricity        | 0.303     | 0.631  | 0.409    | 157     | 0.953    |
| 14 |refugees           | 0.291     | 0.591  | 0.39     | 254     | 0.922    |
| 18 |transport          | 0.307     | 0.478  | 0.374    | 339     | 0.91     |
| 32 |other_weather      | 0.271     | 0.542  | 0.361    | 419     | 0.867    |
| 22 |hospitals          | 0.286     | 0.4    | 0.333    | 85      | 0.977    |
| 17 |infrastructure_rel | 0.234     | 0.5    | 0.319    | 504     | 0.822    |
| 5  |search_and_rescue  | 0.236     | 0.399  | 0.296    | 208     | 0.935    |
| 25 |other_infrastruct  | 0.188     | 0.494  | 0.272    | 346     | 0.848    |
| 13 |missing_people     | 0.221     | 0.352  | 0.271    | 91      | 0.971    |
| 24 |aid_centers        | 0.16      | 0.372  | 0.224    | 86      | 0.963    |
| 21 |tools              | 0.389     | 0.146  | 0.212    | 48      | 0.991    |
| 6  |security           | 0.141     | 0.301  | 0.192    | 123     | 0.948    |
| 1  |offer              | 0.129     | 0.1    | 0.113    | 40      | 0.99     |
| 23 |shops              | 0         | 0      | 0        | 38      | 0.994    |


## Deployment

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
This will start the web app and will direct you to a URL (http://127.0.0.1:5000/) where you can enter messages and get classification results for it.


**File Structure**  
.  
├── app     
│   ├── run.py                           
│   └── templates   
│     ├── go.html                      
│     └── master.html                  
├── data                   
│   ├── disaster_categories.csv          
│   ├── disaster_messages.csv            
│   └── process_data.py                  
├── models  
│   └── train_classifier.py  
│   └── classifier.pkl  
└── README.md  


app folder contains the following HTML iinside `templates` folder:  
  - `master.html`: Renders homepage  
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


## Screenshots:

<p align="center">
  <img src="https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/image/Screenshot%202023-11-23%20024336.png" width="750" title="title">
</p>

<p align="center">
  <img src="https://github.com/TheAnalyticCraft/Disaster-Response-Pipeline-Project/blob/main/image/Screenshot%202023-11-23%20024246.png" width="750" title="title">
</p>

## Acknowledgements

* [Udacity](https://www.udacity.com/) for offering an exceptional Data Engineering Program
* [Figure Eight](https://www.figure-eight.com/) for providing the dataset essential for model training
