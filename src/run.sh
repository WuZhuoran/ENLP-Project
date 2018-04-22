#!/usr/bin/env bash

echo '***************************************'
echo 'Now Checking python and pip version...'
echo '***************************************'

python -c 'import sys; assert sys.version_info[:][0] >= 3; assert sys.version_info[:][1] >= 5'
python -c 'import pip; assert pip.__version__ >= str(10.0)'

echo '***************************************'
echo 'Now Installing requirement package.....'
echo '***************************************'
pip install -r ../requirement.txt

echo '***************************************'
echo 'Now Running Data Analysis..............'
echo '***************************************'
python data_analysis.py

echo '***************************************'
echo 'Now Running Majority Baseline Model....'
echo '***************************************'
python model_majority.py

echo '***************************************'
echo 'Now Running TextBlob Baseline Model....'
echo '***************************************'
python model_xgboost.py

echo '***************************************'
echo 'Now Running Naive Bayes Model..........'
echo '***************************************'
python model_naive_bayes.py

echo '***************************************'
echo 'Now Running Support Vector Machine Model'
echo '***************************************'
python model_svm.py

echo '***************************************'
echo 'Now Running RNN-LSTM Model.............'
echo '***************************************'
python model_lstm.py

echo '***************************************'
echo 'Now Running Ensumble Model.............'
echo '***************************************'
python model_ensemble_data.py
python model_ensemble.py
