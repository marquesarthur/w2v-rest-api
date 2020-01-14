

# Docker Word2Vec REST API

This is a docker image that takes two sentences and returns the cosine similarity between their vector representations.

The model uses the `word2vec-google-news-300` trained dataset. 

A sentence is converted into its vector representation following the algorithm (pesudo-code):

```python
def vector(sentence):
    model = gensim word2vec-google-news-300
    c = 0
    vector = numpy.zeros(model.size) 
    for word in sentence:
        if c in model.vocab:
            vector += model.word
            c += 1
    
    return numpy.divide(vector, c)
            

def similarity(a, b):
    a_vec = vector(a)
    b_vec = vector(b)
    return cosine_similarity(a_vec, b_vec)
```

First, we convert the sentences to their vector representations. Then, we compute the cosine similarity between them.
To convert a sentence, we check whether a word exists in the model vocabulary. If it does, we retrieve the vector representation of the word from the model.
The final vector is computing summing all vectors of the words that exist in the model and diving the number of words that exist in the model.
The division assists in accounting for lengthy sentences. 

The final implementation uses [Aerim Kim](https://www.linkedin.com/in/aerinykim/)'s [python code](https://bitbucket.org/yunazzang/aiwiththebest_byor/src/master/)

## Setup

### From docker hub

```shell script
docker pull msarthur/w2v-rest:latest
```

### From source

```shell script

git clone git@github.com:marquesarthur/w2v-rest-api.git
cd w2v-rest-api
docker build -t msarthur/w2v-rest .
```


## Publish image

```shell script
docker push msarthur/w2v-rest:latest
```

___

## RUN

```shell script
docker run --name w2v-rest -p 5001:5001 msarthur/w2v-rest
```

## Testing
```shell script
curl -XGET 'http://localhost:5001/w2v/similarity?a=Have+you+ever+tried+to+used+the+method+string.split%28%22hello+world%22%29&b=try+using+the+string.split+method' | json_pp
```

Response should look like:

```json
{
   "a" : "Have you ever tried to used the method string.split(\"hello world\")",
   "b" : "try using the string.split method",
   "err" : null,
   "sim" : "0.54513955"
}
``` 

You can also run the [test file](test.py) and test the API programmatically

# More models and implementations in the near future !!!