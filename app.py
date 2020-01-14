from flask import Flask, request, jsonify

print("Loading stopwords")

import nltk
from util import clean_text

nltk.download('stopwords')
nltk.download('punkt')

from embedding import Word2Vec

print("Loading word2vec")
w2v = Word2Vec()
w2v.load(gensim_pre_trained_model="word2vec-google-news-300")

print("Init flask")
app = Flask(__name__)


@app.route("/", methods=['GET'])
def welcome():
    return "Welcome to our Machine Learning REST API!"


@app.route("/w2v/similarity", methods=['GET'])
def similarity_route():
    response = {
        "a": None,
        "b": None,
        "sim": None,
        "err": None
    }
    try:
        a = request.args.get("a")
        b = request.args.get("b")
        y = clean_text(a)
        x = clean_text(b)
        sim = w2v.similarity(x, y)
        response['a'] = a
        response['b'] = b
        response['sim'] = str(sim)
        return jsonify(response)
    except Exception as err:
        response['err'] = str(err)
        return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
