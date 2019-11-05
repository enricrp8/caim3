from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import functools
import math
import numpy as np
import operator
import argparse
import re


index = ""
nrounds = 0
k = 0
R = 0
alpha = 0
beta = 0
query = []

words = {}
operators = {}


def fill_dicc_from_tw(v):
    for w in words:
        words[w] *= alpha

    for t, w in v:
        if t in words:
            words[t] += (beta/k) * w
        else:
            operators[t] = '~'
            words[t] = (beta/k) * w


def search_file_by_path(client, index, path):
    """
    Search for a file using its path

    :param path:
    :return:
    """
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


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

    tfidfwN = normalize(tfidfw)
    return tfidfwN


def normalize(tw):
    sum_squares = functools.reduce(lambda x, y: x + math.pow(y[1], 2), tw, 0)
    norm = np.sqrt(sum_squares)
    tw_normalized = list(map(lambda x: (x[0], x[1]/norm), tw))
    return tw_normalized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default="news", help='Index to search')
    parser.add_argument('--nrounds', default=3, type=int,
                        help='Number of hits to return')
    parser.add_argument('--R', default=6, type=int,
                        help='Number of words to get from each document')
    parser.add_argument('--alpha', default=0.8, type=float, help='alpha')
    parser.add_argument('--beta', default=0.2, type=float, help='beta')
    parser.add_argument('--k', default=8, type=int,
                        help='Number of important documents')
    parser.add_argument('--query', default=[],
                        nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    index = args.index
    nrounds = args.nrounds
    R = args.R
    alpha = args.alpha
    beta = args.beta
    k = args.k
    query = args.query

    if index.strip() != "":
        try:
            client = Elasticsearch()
            s = Search(using=client, index=index)

            queries = []

            for t in query:
                importantW = t.split('^')
                if len(importantW) == 1:
                    fuzzyW = t.split('~')
                    if len(fuzzyW) == 1:
                        words[t] = 1
                        operators[t] = '^'
                        queries.append(t + str(f'^1'))
                    else:
                        words[fuzzyW[0]] = float(fuzzyW[1])
                        operators[fuzzyW[0]] = '~'
                        queries.append(t)
                else:
                    words[importantW[0]] = float(importantW[1])
                    operators[importantW[0]] = '^'
                    queries.append(t)

            if len(query) != 0:

                for i in range(0, nrounds):

                    q = Q('query_string', query=queries[0])
                    for i in range(1, len(queries)):
                        q &= Q('query_string', query=queries[i])
                    s = s.query(q)
                    response = s[0:k].execute()
                    tw = {}
                    for r in response:
                        file_id = r.meta.id
                        file_tw = toTFIDF(client, index, file_id)
                        for (t, w) in file_tw:
                            if t in tw:
                                tw[t] = tw[t] + w
                            else:
                                tw[t] = w
                    twv = [(t, tw[t]) for t in tw]
                    twv.sort(
                        key=operator.itemgetter(1), reverse=True)
                    fill_dicc_from_tw(twv[0:R])
                    queries = [(t + str(f'{operators[t]}{words[t]}'))
                               for t in words]
                    print(queries)
                    print(f"{response.hits.total['value']} Documents")
            else:
                print('No query parameters passed')
        except NotFoundError:
            print(f'Index {index} does not exists')
    else:
        print("No index entered")
