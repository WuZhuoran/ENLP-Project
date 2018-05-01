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

* [@Mitchell Abrams](https://github.com/mjabrams) M.S. Student, Department of Linguistics, Georgetown University <[mja248@georgetown.edu](mailto:mja248@georgetown.edu)>
* [@Shaobo Wang](https://github.com/sw1001) PhD. Student, Department of Computer Science, Georgetown University <[sw1001@georgetown.edu](mailto:sw1001@georgetown.edu)>
* [@Zhuoran Wu](https://github.com/WuZhuoran) M.S. Student, Department of Computer Science, Georgetown University <[zw118@georgetown.edu](mailto:zw118@georgetown.edu)>

## Acknowledge

Special thanks to [@Nathan Schneider](http://people.cs.georgetown.edu/nschneid/) as our instructor of `Empirical Methods in Natural Language Processing `

Special thanks to [@Sean Simpson](http://www.seanskylersimpson.com/) and [@Austin Blodgett](http://www.austinblodgett.name/) as TAs.

## Reference

* Benamara, Farah, Maite Taboada, and Yannick Mathieu. "Evaluative language beyond bags of words: Linguistic insights and computational applications." Computational Linguistics 43.1 (2017): 201-264.
* Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
* Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan, Thumbs up? Sentiment Classification using Machine Learning Techniques, Proceedings of EMNLP 2002.
* Bo Pang and Lillian Lee, A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, Proceedings of ACL 2004.
* McCallum, Andrew, and Kamal Nigam. "A comparison of event models for naive bayes text classification." AAAI-98 workshop on learning for text categorization. Vol. 752. No. 1. 1998.
* Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL, pages 115–124.
* Pedregosa, Fabian, et al. "Scikit-learn: Machine learning in Python." Journal of machine learning research 12.Oct (2011): 2825-2830.
* Snyder, Benjamin, and Regina Barzilay. "Multiple aspect ranking using the good grief algorithm." Human Language Technologies 2007: The Conference of the North American Chapter of the Association for Computational Linguistics; Proceedings of the Main Conference. 2007.
* Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." Proceedings of the 2013 conference on empirical methods in natural language processing. 2013.
* Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient Estimation of Word Representations in Vector Space. In Proceedings of Workshop at ICLR, 2013.
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.
* Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations. In Proceedings of NAACL HLT, 2013.
