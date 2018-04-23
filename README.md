# ENLP-Project

[Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews), 
Project for `COSC-572/LING-572 Spring 2018 @ Georgetown University` **[Empirical Methods in Natural Language Processing](http://people.cs.georgetown.edu/cosc572/s18/schedule.html)**

Task: Classify the sentiment of sentences from the Rotten Tomatoes DataSet.

## Getting Started

### Requirements

```bash
pip install -r requirement.txt
```

For Windows, please see [documentation](http://xgboost.readthedocs.io/en/latest/build.html#building-on-windows) about how to install `XGBoost` on Windows.

Requires `Python 3.5` and `pip 10.0.1` or higher.

## Usage

Run Everything - Estimated Running Time: `32m 50.436s`
```bash
cd src
sh run.sh
```

Run Data Analysis - Estimated Running Time: `220.78s`
```bash
python data_analysis.py
```

Run Majority Baseline Model - Estimated Running Time: `0.066s`
```bash
python model_majority.py
```

Run TextBlob Baseline Model - Estimated Running Time: `14.06s`
```bash
python model_xgboost.py
```

Run Naive Bayes Model - Estimated Running Time: `2.20s`
```bash
python model_naive_bayes.py
```

Run Random Forest Model - Estimated Running Time: `566.64s`
```bash
python model_all.py
```

Run SVM Model - Estimated Running Time: `80.70s`+
```bash
python model_svm.py
```

Run RNN-LSTM Model - Estimated Running Time: `400.96s`
```bash
python model_lstm.py
```

Run Multiclass Model - Estimated Running Time: `645.58s`+
```bash
python model_ensemble_data.py
python model_ensemble.py
```

## Report

[Final Report](https://github.com/sw1001/ENLP-Project/tree/master/report/related_work) is located at report folder.

## Authors

* [@Mitchell Abrams](https://github.com/mjabrams) <[mja248@georgetown.edu](mailto:mja248@georgetown.edu)>
* [@Shaobo Wang](https://github.com/sw1001) <[sw1001@georgetown.edu](mailto:sw1001@georgetown.edu)>
* [@Zhuoran Wu](https://github.com/WuZhuoran) <[zw118@georgetown.edu](mailto:zw118@georgetown.edu)>

## Acknowledge

Special thanks to [@Nathan Schneider](http://people.cs.georgetown.edu/nschneid/) as our instructor of `Empirical Methods in Natural Language Processing `

Special thanks to [@Sean Simpson](http://www.seanskylersimpson.com/) and [@Austin Blodgett](http://www.austinblodgett.name/) as TA.

## Reference
