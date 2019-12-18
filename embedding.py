# the following codes aree packaged as a py file that can be import in other python scripts.
# this user own package is named as "data_prepare"


import logging
import time

import gensim.downloader as api
import numpy as np
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


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

        x_vector = self.avg_sentence_vector(x)
        y_vector = self.avg_sentence_vector(y)

        result = cosine_similarity(x_vector.reshape(1, -1), y_vector.reshape(1, -1))
        return result[0].tolist()[0]  # there must be a better way to do this

    def avg_sentence_vector(self, sentence):
        """
        Obtains the vector representation of a sentence.
        A vector representation of a sentence is computed based on the sum of all the words in the sentence that exists in the model vocabulary.
        The vector is then normalized dividing the summation by the number of words in the sentence.

        :param sentence: sentence containing words to be converted into a vector
        :return: vectorized representation of the sentence
        """

        n_features = self.model.vector_size
        sentence_vec = np.zeros((n_features,), dtype="float32")

        nwords = 0
        for word in word_tokenize(sentence):
            if word not in self.model:
                sentence_vec = np.add(sentence_vec, np.zeros((n_features,), dtype="float32"))
            else:
                sentence_vec = np.add(sentence_vec, self.model[word])

            nwords += 1

        if nwords > 0:
            sentence_vec = np.divide(sentence_vec, nwords)

        return sentence_vec

    def load(self, gensim_pre_trained_model="word2vec"): # fasttext-wiki-news-subwords-300
        start = time.time()
        logging.info("Loading model")
        self.model = api.load(gensim_pre_trained_model)
        end = time.time()
        logging.info(">> model loaded")
        logging.info(">> %s" % (end - start))
