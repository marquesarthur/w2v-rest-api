import os

from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
from gensim.utils import tokenize
from gensim import utils


input_corpus = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sample.txt')


class MyIter:

    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        # path = datapath()
        with utils.open(self.input_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                yield list(tokenize(line))

# https://stackoverflow.com/questions/67573416/unable-to-recreate-gensim-docs-for-training-fasttext-typeerror-either-one-of-c
model = FastText(vector_size=4, window=3, min_count=1)
model.build_vocab(MyIter(input_corpus))
total_examples = model.corpus_count

print(model.corpus_count)
model.train(MyIter(input_corpus), total_examples=total_examples, epochs=5)


print(model.wv[' '])
print(model.wv['use'])
print(model.wv['even'])


# for tokens in MyIter(input_corpus):
#     print(tokens)

