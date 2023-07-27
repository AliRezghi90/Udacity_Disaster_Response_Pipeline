<p align="center">
  <a href="https://www.udacity.com/">
    <img src='https://course_report_production.s3.amazonaws.com/rich/rich_files/rich_files/5511/s300/udacity-logo.png' alt="Udacity logo" width = 100px>
   </a>
</p>
<h3 align="center"><a href='https://www.udacity.com/course/data-scientist-nanodegree--nd025'>Udacity Data Scientist Nanodegree Program</a></h3>
<h1 align="center"> Disaster Response Pipeline </h1>


The project files can be found [here](https://github.com/AliRezghi90/Udacity_Disaster_Response_Pipeline.git) 

## Table of Contents
- [Introduction](#introduction)
- [Installation - Packages](#installation)
- [File Descriptions](#files)
- [Licensing, Authors, and Acknowledgements](#licensing)


## Introduction <a name="introduction"></a>
During emergency events such as earthquake or flooding, thousands of messages would be delivered to different disaster response agencies. [Figure Eight](https://www.figure-eight.com/) gathered and prepraded pre-labeled data for over 26000 messages (Figure Eight was acquired by [Appen](https://appen.com/) in 2019). 

Using this database, a disaster response web application is created to identify the correct category for real-time messages received during a disaster. Accordingly, disaster response agencie can take appropriate actions as quickly as possible.


## Installation <a name="installation"></a>
Use pip to install The following packages:

- Numpy
- Pandas
- Flask
- sqlalchemy
- nltk
- sklearn

The program was appropriately run using Python 3.11., however, older versions might work as well.


## File Descriptions <a name="files"></a>

#### "data" folder:
* **process_data.py**: This python excutuble code applies the extract, transform, and load (ETL) pipeline on the csv files containing message data and categories. The cleaned data is stored in a SQL database named "DisasterResponse.db"
* **disaster_categories**: csv file containing the categories (labels)
* **disaster_messages**: csv file containing the messages
* **ETL Pipeline Preparation.ipynb** The Jupyter notebook that is used to develop process_data.py code

#### "models" folder:
* **train_classifier.py**: This python excutuble code applies the machine learning (ML) pipeline to find the best classifier. 
* **ML Pipeline Preparation.ipynb**: The Jupyter notebook that is used to develop train_classifier.py code

#### "app" folder: 
* **run.py**: The python executuble code to load the web application
* **templates**: folder containing the templates for the web app

#### "Scrennshot" folder: 
Contains the screenshots of the web app and the terminal commands/outputs

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Credit must be given to [Udacity](https://www.udacity.com/) for the excellent Udacity Data Scientist Nanodegree program and also [Figure Eight](https://www.figure-eight.com/) for giving access to the pre-labeled datasets.




