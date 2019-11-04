from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from elasticsearch.client import CatClient
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

import functools
import math
import numpy as np
import operator


nrounds = 3
k = 3
R = 10
alpha = 0.8
beta = 0.2

words = {}


def fill_dicc_from_tw(v):
    for w in words:
        words[w] *= alpha

    for t, w in v:
        if t in words:
            words[t] += (beta/k) * w
        else:
            words[t] = (beta/k) * w


def dico_search(x, v, l, r):
    if r >= l:
        mid = int(l + (r - l)/2)
        if(v[mid][0] == x):
            return mid
        elif(v[mid][0] > x):
            return dico_search(x, v, l, mid-1)
        else:
            return dico_search(x, v, mid+1, r)
    else:
        return -1


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
    searchIndex = "news"

    if searchIndex.strip() != "":

        # queries = input("Write the words to query:\n").split(' ')
        queries = ['toronto']

        try:
            client = Elasticsearch()
            s = Search(using=client, index=searchIndex)

            if len(queries) != 0:
                for i in range(0, nrounds):
                    q = Q('query_string', query=queries[0])
                    for i in range(1, len(queries)):
                        q &= Q('query_string', query=queries[i])

                    s = s.query(q)
                    response = s[0:k].execute()
                    tw = []
                    for r in response:
                        file_id = r.meta.id
                        file_tw = toTFIDF(client, searchIndex, file_id)
                        for (t, w) in file_tw:
                            index = dico_search(t, tw, 0, len(tw) - 1)
                            if index == -1:
                                tw.append((t, w))
                            else:
                                tw[index] = (t, tw[index][1] + w)
                    tw.sort(key=operator.itemgetter(1), reverse=True)
                    fill_dicc_from_tw(tw[0:R])
                    queries = [w for w in words]
                    print(queries)
            else:
                print('No query parameters passed')

            print(f"{response.hits.total['value']} Documents")

        except NotFoundError:
            print(f'Index {index} does not exists')
    else:
        print("No index entered")
