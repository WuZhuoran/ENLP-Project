import os
import urllib.request
import nltk
import csv
import random
import numpy
import re
import time
import json
from sklearn.multiclass import OneVsOneClassifier
from collections import namedtuple
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import accuracy_score


DATA_PATH = '../data/'
Datapoint = namedtuple("Datapoint", "phraseid sentenceid phrase sentiment")


def _iter_data_file(filename):
    path = os.path.join(DATA_PATH, filename)
    it = csv.reader(open(path, "r"), delimiter="\t")
    row = next(it)  # Drop column names
    if " ".join(row[:3]) != "PhraseId SentenceId Phrase":
        raise ValueError("Input file has wrong column names: {}".format(path))
    for row in it:
        if len(row) == 3:
            row += (None,)
        yield Datapoint(*row)


def iter_corpus(__cached=[]):
    """
    Returns an iterable of `Datapoint`s with the contents of train.tsv.
    """
    if not __cached:
        __cached.extend(_iter_data_file("train.tsv"))
    return __cached


def iter_test_corpus():
    """
    Returns an iterable of `Datapoint`s with the contents of test.tsv.
    """
    return list(_iter_data_file("test.tsv"))


def make_train_test_split(seed, proportion=0.9):
    """
    Makes a randomized train/test split of the train.tsv corpus with
    `proportion` fraction of the elements going to train and the rest to test.
    The `seed` argument controls a shuffling of the corpus prior to splitting.
    The same seed should always return the same train/test split and different
    seeds should always provide different train/test splits.
    Return value is a (train, test) tuple where train and test are lists of
    `Datapoint` instances.
    """
    data = list(iter_corpus())
    ids = list(sorted(set(x.sentenceid for x in data)))
    if len(ids) < 2:
        raise ValueError("Corpus too small to split")
    N = int(len(ids) * proportion)
    if N == 0:
        N += 1
    rng = random.Random(seed)
    rng.shuffle(ids)
    test_ids = set(ids[N:])
    train = []
    test = []
    for x in data:
        if x.sentenceid in test_ids:
            test.append(x)
        else:
            train.append(x)
    return train, test


def cross_validation(factory, seed, K=10, callback=None):
    seed = str(seed)
    scores = []
    for k in range(K):
        train, test = make_train_test_split(seed + str(k))
        predictor = factory()
        predictor.fit(train)
        score = predictor.score(test)
        if callback:
            callback(score)
        scores.append(score)
    return sum(scores) / len(scores)


def fix_json_dict(config):
    new = {}
    for key, value in config.items():
        if isinstance(value, dict):
            value = fix_json_dict(value)
        elif isinstance(value, str):
            if value == "true":
                value = True
            elif value == "false":
                value = False
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
        new[key] = value
    return new


class PrintPartialCV:
    def __init__(self):
        self.last = time.time()
        self.i = 0

    def report(self, score):
        new = time.time()
        self.i += 1
        print("individual {}-th fold score={}% took {} seconds".format(self.i, score * 100, new - self.last))
        self.last = new


class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self


class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a samr pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)


class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements should be a list of `(from, to)` tuples of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets(StatelessTransform):
    """
    This transformation replaces words in the input with their Wordnet
    synsets[0].
    The intuition behind it is that phrases represented by synset vectors
    should be "closer" to one another (not suffer the curse of dimensionality)
    than the sparser (often poetical) words used for the reviews.
    [0] For example "bank": http://wordnetweb.princeton.edu/perl/webwn?s=bank
    """
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        containins the names of the synsets of all the words in `X[i]`,
        excluding noun synsets.
        `X[i]` is internally tokenized using `str.split`, so it should be
        formatted accordingly.
        """
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            ss = nltk.wordnet.wordnet.synsets(word)
            result.extend(str(s) for s in ss if ".n." not in str(s))
        return " ".join(result)


class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()


class ClassifierOvOAsFeatures:
    """
    A transformation that esentially implement a form of dimensionality
    reduction.
    This class uses a fast SGDClassifier configured like a linear SVM to produce
    a vector of decision functions separating target classes in a
    one-versus-rest fashion.
    It's useful to reduce the dimension bag-of-words feature-set into features
    that are richer in information.
    """
    def fit(self, X, y):
        """
        `X` is expected to be an array-like or a sparse matrix.
        `y` is expected to be an array-like containing the classes to learn.
        """
        self.classifiers = OneVsOneClassifier(SGDClassifier(), n_jobs=-1).fit(X, numpy.array(y)).estimators_
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be an array-like or a sparse matrix.
        It returns a dense matrix of shape (n_samples, m_features) where
            m_features = (n_classes * (n_classes - 1)) / 2
        """
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)


FIELDS = ("Entry, Source, Positiv, Negativ, Pstv, Affil, Ngtv, Hostile, Strong,"
          " Power, Weak, Submit, Active, Passive, Pleasur, Pain, Feel, Arousal,"
          " EMOT, Virtue, Vice, Ovrst, Undrst, Academ, Doctrin, Econ, Exch, "
          "ECON, Exprsv, Legal, Milit, Polit, POLIT, Relig, Role, COLL, Work, "
          "Ritual, SocRel, Race, Kin, MALE, Female, Nonadlt, HU, ANI, PLACE, "
          "Social, Region, Route, Aquatic, Land, Sky, Object, Tool, Food, "
          "Vehicle, BldgPt, ComnObj, NatObj, BodyPt, ComForm, COM, Say, Need, "
          "Goal, Try, Means, Persist, Complet, Fail, NatrPro, Begin, Vary, "
          "Increas, Decreas, Finish, Stay, Rise, Exert, Fetch, Travel, Fall, "
          "Think, Know, Causal, Ought, Perceiv, Compare, Eval, EVAL, Solve, "
          "Abs, ABS, Quality, Quan, NUMB, ORD, CARD, FREQ, DIST, Time, TIME, "
          "Space, POS, DIM, Rel, COLOR, Self, Our, You, Name, Yes, No, Negate, "
          "Intrj, IAV, DAV, SV, IPadj, IndAdj, PowGain, PowLoss, PowEnds, "
          "PowAren, PowCon, PowCoop, PowAuPt, PowPt, PowDoct, PowAuth, PowOth, "
          "PowTot, RcEthic, RcRelig, RcGain, RcLoss, RcEnds, RcTot, RspGain, "
          "RspLoss, RspOth, RspTot, AffGain, AffLoss, AffPt, AffOth, AffTot, "
          "WltPt, WltTran, WltOth, WltTot, WlbGain, WlbLoss, WlbPhys, WlbPsyc, "
          "WlbPt, WlbTot, EnlGain, EnlLoss, EnlEnds, EnlPt, EnlOth, EnlTot, "
          "SklAsth, SklPt, SklOth, SklTot, TrnGain, TrnLoss, TranLw, MeansLw, "
          "EndsLw, ArenaLw, PtLw, Nation, Anomie, NegAff, PosAff, SureLw, If, "
          "NotLw, TimeSpc, FormLw, Othtags, Defined")

InquirerLexEntry = namedtuple("InquirerLexEntry", FIELDS)
FIELDS = InquirerLexEntry._fields


class InquirerLexTransform(StatelessTransform):
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ IAV Strong".split()]

    def transform(self, X, y=None):
        """
        `X` is expected to be a list of `str` instances containing the phrases.
        Return value is a list of `str` containing different amounts of the
        words "Positiv_Positiv", "Negativ_Negativ", "IAV_IAV", "Strong_Strong"
        based on the sentiments given to the input words by the Hardvard
        Inquirer lexicon.
        """
        corpus = self._get_corpus()
        result = []
        for phrase in X:
            newphrase = []
            for word in phrase.split():
                newphrase.extend(corpus.get(word.lower(), []))
            result.append(" ".join(newphrase))
        return result

    def _get_corpus(self):
        """
        Private method used to cache a dictionary with the Harvard Inquirer
        corpus.
        """
        if not self._corpus:
            corpus = defaultdict(list)
            it = csv.reader(open(os.path.join(DATA_PATH, "inquirerbasicttabsclean")),
                            delimiter="\t")
            next(it)  # Drop header row
            for row in it:
                entry = InquirerLexEntry(*row)
                xs = []
                for i in self._use_fields:
                    name, x = FIELDS[i], entry[i]
                    if x:
                        xs.append("{}_{}".format(name, x))
                name = entry.Entry.lower()
                if "#" in name:
                    name = name[:name.index("#")]
                corpus[name].extend(xs)
            self._corpus.append(dict(corpus))
        return self._corpus[0]


_valid_classifiers = {
    "sgd": SGDClassifier,
    "knn": KNeighborsClassifier,
    "svc": SVC,
    "randomforest": RandomForestClassifier,
}


def target(phrases):
    return [datapoint.sentiment for datapoint in phrases]


class PhraseSentimentPredictor:
    """
    Main `samr` class. It implements a trainable predictor for phrase
    sentiments. API is a-la scikit-learn, where:
        - `__init__` configures the predictor
        - `fit` trains the predictor from data. After calling `fit` the instance
          methods should be free side-effect.
        - `predict` generates sentiment predictions.
        - `score` evaluates classification accuracy from a test set.
    Outline of the predictor pipeline is as follows:
    A configurable main classifier is trained with a concatenation of 3 kinds of
    features:
        - The decision functions of set of vanilla SGDClassifiers trained in a
          one-versus-others scheme using bag-of-words as features.
        - (Optionally) The decision functions of set of vanilla SGDClassifiers
          trained in a one-versus-others scheme using bag-of-words on the
          wordnet synsets of the words in a phrase.
        - (Optionally) The amount of "positive" and "negative" words in a phrase
          as dictated by the Harvard Inquirer sentiment lexicon
    Optionally, during prediction, it also checks for exact duplicates between
    the training set and the train set.    """
    def __init__(self, classifier="sgd", classifier_args=None, lowercase=True,
                 text_replacements=None, map_to_synsets=False, binary=False,
                 min_df=0, ngram=1, stopwords=None, limit_train=None,
                 map_to_lex=False, duplicates=False):
        """
        Parameter description:
            - `classifier`: The type of classifier used as main classifier,
              valid values are "sgd", "knn", "svc", "randomforest".
            - `classifier_args`: A dict to be passed as arguments to the main
              classifier.
            - `lowercase`: wheter or not all words are lowercased at the start of
              the pipeline.
            - `text_replacements`: A list of tuples `(from, to)` specifying
              string replacements to be made at the start of the pipeline (after
              lowercasing).
            - `map_to_synsets`: Whether or not to use the Wordnet synsets
              feature set.
            - `binary`: Whether or not to count words in the bag-of-words
              representation as 0 or 1.
            - `min_df`: Minumim frequency a word needs to have to be included
              in the bag-of-word representation.
            - `ngram`: The maximum size of ngrams to be considered in the
              bag-of-words representation.
            - `stopwords`: A list of words to filter out of the bag-of-words
              representation. Can also be the string "english", in which case
              a default list of english stopwords will be used.
            - `limit_train`: The maximum amount of training samples to give to
              the main classifier. This can be useful for some slow main
              classifiers (ex: svc) that converge with less samples to an
              optimum.
            - `max_to_lex`: Whether or not to use the Harvard Inquirer lexicon
              features.
            - `duplicates`: Whether or not to check for identical phrases between
              train and prediction.
        """
        self.limit_train = limit_train
        self.duplicates = duplicates

        # Build pre-processing common to every extraction
        pipeline = [ExtractText(lowercase)]
        if text_replacements:
            pipeline.append(ReplaceText(text_replacements))

        # Build feature extraction schemes
        ext = [build_text_extraction(binary=binary, min_df=min_df,
                                     ngram=ngram, stopwords=stopwords)]
        if map_to_synsets:
            ext.append(build_synset_extraction(binary=binary, min_df=min_df,
                                               ngram=ngram))
        if map_to_lex:
            ext.append(build_lex_extraction(binary=binary, min_df=min_df,
                                            ngram=ngram))
        ext = make_union(*ext)
        pipeline.append(ext)

        # Build classifier and put everything togheter
        if classifier_args is None:
            classifier_args = {}
        classifier = _valid_classifiers[classifier](**classifier_args)
        self.pipeline = make_pipeline(*pipeline)
        self.classifier = classifier

    def fit(self, phrases, y=None):
        """
        `phrases` should be a list of `Datapoint` instances.
        `y` should be a list of `str` instances representing the sentiments to
        be learnt.
        """
        y = target(phrases)
        if self.duplicates:
            self.dupes = DuplicatesHandler()
            self.dupes.fit(phrases, y)
        Z = self.pipeline.fit_transform(phrases, y)
        if self.limit_train:
            self.classifier.fit(Z[:self.limit_train], y[:self.limit_train])
        else:
            self.classifier.fit(Z, y)
        return self

    def predict(self, phrases):
        """
        `phrases` should be a list of `Datapoint` instances.
        Return value is a list of `str` instances with the predicted sentiments.
        """
        Z = self.pipeline.transform(phrases)
        labels = self.classifier.predict(Z)
        if self.duplicates:
            for i, phrase in enumerate(phrases):
                label = self.dupes.get(phrase)
                if label is not None:
                    labels[i] = label
        return labels

    def score(self, phrases):
        """
        `phrases` should be a list of `Datapoint` instances.
        Return value is a `float` with the classification accuracy of the
        input.
        """
        pred = self.predict(phrases)
        return accuracy_score(target(phrases), pred)

    def error_matrix(self, phrases):
        predictions = self.predict(phrases)
        matrix = defaultdict(list)
        for phrase, predicted in zip(phrases, predictions):
            if phrase.sentiment != predicted:
                matrix[(phrase.sentiment, predicted)].append(phrase)
        return matrix


def build_text_extraction(binary, min_df, ngram, stopwords):
    return make_pipeline(CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram),
                                         stop_words=stopwords),
                         ClassifierOvOAsFeatures())


def build_synset_extraction(binary, min_df, ngram):
    return make_pipeline(MapToSynsets(),
                         CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram)),
                         ClassifierOvOAsFeatures())


def build_lex_extraction(binary, min_df, ngram):
    return make_pipeline(InquirerLexTransform(),
                         CountVectorizer(binary=binary,
                                         tokenizer=lambda x: x.split(),
                                         min_df=min_df,
                                         ngram_range=(1, ngram)),
                         Densifier())


class DuplicatesHandler:
    def fit(self, phrases, target):
        self.dupes = {}
        for phrase, label in zip(phrases, target):
            self.dupes[self._key(phrase)] = label

    def get(self, phrase):
        key = self._key(phrase)
        return self.dupes.get(key)

    def _key(self, x):
        return " ".join(x.phrase.lower().split())


class _Baseline:
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return ["2" for _ in X]

    def score(self, X):
        gold = target(X)
        pred = self.predict(X)
        return accuracy_score(gold, pred)


if __name__ == '__main__':
    filename = DATA_PATH + 'model.json'
    config = json.load(open(filename))

    factory = lambda: PhraseSentimentPredictor(**config)
    factory()  # Run once to check config is ok

    report = PrintPartialCV()
    result = cross_validation(factory, seed="robot rock", callback=report.report)

    print("10-fold cross validation score {}%".format(result * 100))
