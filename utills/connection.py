from pymongo import MongoClient

def open_connection():
    #client = MongoClient('mongodb://datastore:27017',connect=False)
    client = MongoClient('mongodb://127.0.0.1:27017')
    return client