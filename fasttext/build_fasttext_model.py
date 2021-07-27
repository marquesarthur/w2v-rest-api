import os

from gensim.models.fasttext import FastText, save_facebook_model, load_facebook_model
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils


# hyperparameters
window=3
min_count=1
vector_size=4

# input paths
input_corpus = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample.txt')
model_bin = os.path.join(os.path.dirname(os.path.realpath(__file__)), "SO_fasttext_vectors_200.bin")

class MyIter:

    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        # path = datapath()
        with utils.open(self.input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))

# https://stackoverflow.com/questions/67573416/unable-to-recreate-gensim-docs-for-training-fasttext-typeerror-either-one-of-c
# FIXME: enable once sample is fully finished
# model = FastText(vector_size=vector_size, window=window, min_count=min_count)
# print('build vocab...')
# model.build_vocab(MyIter(input_corpus))
# total_examples = model.corpus_count
#
# print(model.corpus_count)
# print('train model...')
# model.train(MyIter(input_corpus), total_examples=total_examples, epochs=5)
#
#
# print(model.wv[' '])
# print(model.wv['use'])
# print(model.wv['even'])
#
# print('saving model...')
# save_facebook_model(model, model_bin)


# load model?
print('loading saved model...')
fb_model = load_facebook_model(model_bin)
print(fb_model.wv[' '])
print(fb_model.wv['use'])
print(fb_model.wv['even'])
