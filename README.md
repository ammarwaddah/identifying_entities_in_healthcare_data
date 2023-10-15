# Identifying Entities in Healthcare Data
Using Machine Learning, Data Science and NLP techniques to identifying entities in healthcare data using significant features given by the most linked features that extraction from that are taken into consideration when evaluating the entities of healthcare data.

## Table of Contents
* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Observation](#observation)
* [Approach](#Approach)
* [Dataset General info](#dataset-general-info)
* [Evaluation](#evaluation)
* [Technologies](#technologies)
* [Setup](#setup)
* [Features](#features)
* [Run Example](#run-example)
* [Sources](#sources)

## Introduction
The field of Natural Language Processing (NLP) has introduced several effective and important applications in the real life, and its results have yielded many applications, including Named Entity Recognition (NER) applications and played a major role in the medical field by identifying diseases and determining treatment methods by extracting information from large amounts of data that contain diseases and treatment methods, and using Artificial Intelligence to distinguish them through pre-processing methods after appropriate analysis of the data and then optimal implementation of Artificial Intelligence models in the Named Entity Recognition (NER).
Hence, I present to you my Named Entity Recognition project (Identifying Entities in Healthcare Data). In this project I put my suggestions for solving it with the best possible ways and the current capabilities using Machine Learning, DatSa Science, and NLP.\
Hoping to improve it gradually in the coming times.

## Problem Statement
A health tech company called BeHealthy aimed to connect the medical communities with millions of patients across the country.
BeHealthy has a web platform that allows doctors to list their services and manage patient interactions and provides services for patients such as booking interactions with doctors and ordering medicine online. Here, doctors can easily organise appointments, track past medical records and provide e-prescriptions.

BeHealthy require predictive model which can identify disease and treatment from the patient interaction with doctors or ordering medicine online.

## Observation
The idea of this task is to extract the useful information from text, and in addition to that, the entity of the useful information also need to be identified.
Because the information we are planning to extract from the statement interactions between doctor and patients related to medical terms which are not regular usage day-to-day words.

## Approach
By observing the requirement, it is clearly visible that we have to process the textual sentence and identify the entities like Disease and Treatment. We can predict these all requirements using - CRF (Conditional Random Field) classifier - HMM (Hidden Markov Model) - Perceptron Model - SGD Classifier Model - MultinomialNB Classifier Model - Passive Aggressive Classifier Model - Transformers (I am trying to adjust the data to use it with the best results).

## Dataset General info
**General info about the dataset:**
* About:

We have four data file for this activity to proceed, they are:

1. Train Sentence Dataset.
2. Train Label Dataset.
3. Test Sentence Dataset.
4. Test Label Dataset.

Sentence file contains all interactions between patients and doctor, and Label file contains all entity tags for particular words arranged as per sentence.
We need to do few preprocessing while accessing the dataset we will explore that in further steps.

## Evaluation

1. Value Prediction.
2. Cross-Validation.
3. Classification report.
4. Accuracy Score.
5. Confusion Matrix.

## Technologies
* Programming language: Python.
* Libraries: pycrf, sklearn-crfsuite, eli5, numpy, python-math, more-itertools, regex, matplotlib, pandas, seaborn, nltk, scikit-learn, scipy, random, os, tqdm.
* Application: Jupyter Notebook.

## Setup
To run this project, setup the following libraries on your local machine using pip on the terminal after installing Python:

'''\
pip install pycrf\
pip install sklearn-crfsuite\
pip install eli5\
pip install numpy\
pip install python-math\
pip install regex\
pip install matplotlib\
pip install pandas\
pip install seaborn\
pip install nltk\
pip install scikit-learn\
pip install scipy\
pip install random2\
pip install tqdm\

'''\
To install these packages with conda run:\
'''\
conda install -c conda-forge pycraf\
conda install -c conda-forge sklearn-crfsuite\
conda install -c conda-forge eli5\
conda install -c anaconda numpy\
conda install -c conda-forge mpmath\
conda install -c conda-forge re2\
conda install -c conda-forge matplotlib\
conda install -c anaconda pandas\
conda install -c anaconda seaborn\
conda install -c anaconda nltk\
conda install -c anaconda scikit-learn\
conda install -c anaconda scipy\
conda install -c conda-forge mkl_random\
conda install -c conda-forge tqdm\

'''

## Features
* I present to you my project solving the problem of named entity recognition of healthcare data using a lot of effective algorithms and techniques with a good analysis (EDA), and comparing between them using logical thinking, and putting my suggestions for solving it in the best possible ways and the current capabilities using Data Science, Machine Learning and NLP.

### To Do:
**Briefly about the process of the project work, here are (some) insights that I took care of it:**

* Planning.
* Exploring data.
* Exploratory data analysis (EDA).
* Data Cleaning and PreProcessing.
* Deep EDA (POS, NER).
* Vectorizing (Dict Vectorizer).
* Modelling (CRF, Perceptron, SVC, XGBoost, SGD, Voting, Multinomial NB, Gradient Boosting, Logistic Regression, Ridge, Passive Aggressive, KNN classifiers).
* Evaluating and making analysis of ML models (Confusion Matrix, Classification report, Cross-validation, Accuracy Score, and Value Prediction).
* Parameters Choosing (Halving Grid Search CV).

## Run Example

To run and show analysis, insights, correlation, and results between any set of features of the dataset, here is a simple example of it:

* Note: you have to use a jupyter notebook to open this code file.

1. Run the importing stage.
2. Load the dataset.
3. Select which cell you would like to run and show its output.
4. Run Selection/Line in Python Terminal command (Shift+Enter).

## Sources
This data was taken from Kaggle:\
https://www.kaggle.com/datasets/arunagirirajan/medical-entity-recognition-ner
