from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q

nrounds = 3
k = 1
R = 10
alpha = 1232
beta = 1231

if __name__ == '__main__':
    index = input("Index to serach:\n")

    if index.strip() != "":

        queries = input("Write the words to query:\n").split(' ')

        nhits = input("Enter the number of hits:\n")

        nhits = 1 if nhits.strip() == "" else int(nhits)

        try:
            client = Elasticsearch()
            s = Search(using=client, index=index)

            if len(queries) != 0:
                q = Q('query_string', query=queries[0])
                for i in range(1, len(queries)):
                    q &= Q('query_string', query=queries[i])

                s = s.query(q)
                response = s[0:nhits].execute()
                for r in response:  # only returns a specific number of results
                    print(f'ID= {r.meta.id} SCORE={r.meta.score}')
                    # print(f"{response.hits.total['value']} Documents")
                    print(f'PATH= {r.path}')

                    print(f'TEXT: {r.text[:50]}')
                    print(
                        '-----------------------------------------------------------------')

            else:
                print('No query parameters passed')

            print(f"{response.hits.total['value']} Documents")

        except NotFoundError:
            print(f'Index {index} does not exists')
    else:
        print("No index entered")
