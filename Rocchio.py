from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import functools
import math
import numpy as np

nrounds = 3
k = 300
R = 10
alpha = 0.8
beta = 0.2

words = {}


def get_query_from_tw(v):
    for w in words:
        words[w] *= alpha

    for t, w in v:
        if t in words:
            words[t] += beta * w
        else:
            words[t] = beta * w

    q = Q('query_string', query="")
    for w in words:
        q &= Q('query_string', query=f'{w}^{words[w]}')


def doc_count(client, index):
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


def document_term_vector(client, index, id):
    termvector = client.termvectors(index=index, id=id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


def calculateTF(w, max_freq):
    return w/max_freq


def calculateIDF(df, dcount):
    return np.log2(dcount/df)


def calculateW(w, max_freq, df, dcount):
    return calculateTF(w, max_freq) * calculateIDF(df, dcount)


def toTFIDF(client, index, file_id):
    # Get document terms frequency and overall terms document frequency
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)

    tfidfw = []
    for (t, w), (_, df) in zip(file_tv, file_df):
        tfidfw.append((t, calculateW(w, max_freq, df, dcount)))

    return normalize(tfidfw)


def normalize(tw):
    sum_squares = functools.reduce(lambda x, y: x + math.pow(y[1], 2), tw, 0)
    norm = np.sqrt(sum_squares)
    tw_normalized = list(map(lambda x: (x[0], x[1]/norm), tw))
    return tw_normalized


if __name__ == '__main__':
    # index = input("Index to serach:\n")
    # index = "news"

    # if index.strip() != "":

    #     # queries = input("Write the words to query:\n").split(' ')
    #     queries = ['toronto']

    #     try:
    #         client = Elasticsearch()
    #         s = Search(using=client, index=index)

    #         if len(queries) != 0:
    #             q = Q('query_string', query=queries[0])
    #             for i in range(1, len(queries)):
    #                 q &= Q('query_string', query=queries[i])

    #             s = s.query(q)
    #             response = s[0:k].execute()
    #             for r in response:
    #                 file_id = r.meta.id

    #                 file_tw = toTFIDF(client, index, file_id)

    #         else:
    #             print('No query parameters passed')

    #         print(f"{response.hits.total['value']} Documents")

    #     except NotFoundError:
    #         print(f'Index {index} does not exists')
    # else:
    #     print("No index entered")
