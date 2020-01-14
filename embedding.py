# the following codes aree packaged as a py file that can be import in other python scripts.
# this user own package is named as "data_prepare"


import logging
import time

import gensim.downloader as api
import numpy as np
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

import numpy as np
import math
from scipy.spatial import distance

from random import sample
import sys
from nltk.corpus import stopwords


def ConvertVectorSetToVecAverageBased(vectorSet, ignore=None):
    if ignore is None:
        ignore = []

    if len(ignore) == 0:
        return np.mean(vectorSet, axis=0)
    else:
        return np.dot(np.transpose(vectorSet), ignore) / sum(ignore)


def PhraseToVec(phrase, model=None):
    if not model:
        raise Exception("No pre-trained model defined")
    cachedStopWords = stopwords.words("english")
    phrase = phrase.lower()
    wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
    vectorSet = []
    for aWord in wordsInPhrase:
        try:
            wordVector = model[aWord]
            vectorSet.append(wordVector)
        except:
            pass
    return ConvertVectorSetToVecAverageBased(vectorSet)


# https://bitbucket.org/yunazzang/aiwiththebest_byor/src/master/PhraseSimilarity.py
class PhraseVector(object):
    def __init__(self, phrase, model=None):
        self.vector = PhraseToVec(phrase, model=model)

    def similarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (
                    np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity



class Word2Vec(object):

    def __init__(self):
        self.model = None

    def similarity(self, x, y):
        """
        Computes the cosine similarity of two sentences.
        First, each sentence is converted into its normalized length vector representation.
        Then, the cosine similarity between sentences vectors are computed.
        Uses sklearn cosine similarity function, which works with both dense and sparse vectors.

        For more details, see:
            https://datascience.stackexchange.com/questions/23969/sentence-similarity-prediction
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
        """
        if not self.model:
            raise Exception("You cannot use a similarity function without a trained model")

        x_vector = PhraseVector(x, model=self.model)
        y_vector = PhraseVector(y, model=self.model)

        result = x_vector.similarity(y_vector.vector)
        return result

    def load(self, gensim_pre_trained_model="word2vec"): # fasttext-wiki-news-subwords-300
        start = time.time()
        logging.info("Loading model")
        self.model = api.load(gensim_pre_trained_model)
        end = time.time()
        logging.info(">> model loaded")
        logging.info(">> %s" % (end - start))
