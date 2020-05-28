import requests
import json

r = requests.get("http://localhost:5001/")

if r.content != b'Welcome to our Machine Learning REST API!':
    raise Exception("Perhaps the server is not running?")

#
payload = {
    'b': "The application needs to obtain the Bugzilla's custom fields and values such that it can use them to query and aggregate bugs based on the custom fields and their values.",
    'a': 'Some methods do not require you to log in.'
}
r = requests.get("http://localhost:5001/w2v/similarity", params=payload)
print(r.content)



r = requests.get("http://localhost:5001/")

if r.content != b'Welcome to our Machine Learning REST API!':
    raise Exception("Perhaps the server is not running?")

#
payload = {
    'txt': "The application needs to obtain the Bugzilla's custom fields and values such that it can use them to query and aggregate bugs based on the custom fields and their values."
}
r = requests.get("http://localhost:5001/w2v/embeddings", params=payload)
print(r.content)