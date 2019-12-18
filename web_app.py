from flask import Flask, request, jsonify


import nltk
nltk.download('stopwords')


from embedding import Word2Vec
from nltk.corpus import stopwords
import string

from nltk.tokenize import word_tokenize


def clean_text(text):
    for c in string.punctuation:
        if c not in ["'", "′", "’"]:
            text = text.replace(c, "")
    text = text.lower().replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("'ll", " will")
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]  # remove non-alphabatical words
    stops = set(stopwords.words("english"))
    word_filter = [w for w in words if not w in stops and len(w) >= 3]
    text_return = " ".join(word_filter)

    for c in string.punctuation:
        text_return = text.replace(c, "")

    return text_return



w2v = Word2Vec()
w2v.load(gensim_pre_trained_model="word2vec-google-news-300")





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
        response['sim'] = sim
        return jsonify(response)
    except Exception as err:
        response['err'] = err.message
        return jsonify(response)



if __name__ == "__main__":
    app.run(port=5000)

