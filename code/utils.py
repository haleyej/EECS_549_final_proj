import os
import csv
import json 

from collections import defaultdict


def load_docs(path: str, docid_key:str = 'docid', text_key:str = 'text') -> list[tuple[int, str]]:
    '''
    Reads in files in jsonl format, returns list of tuples containing the docid and text
    '''
    docs = []
    with open(path) as f: 
        doc = f.readline()
        while doc:
            doc = json.loads(doc)
            docid = doc[docid_key]
            text = doc[text_key]
            docs.append((docid, text))
            doc = f.readline()
    return docs


def load_true_relevance(file: str) -> dict[int, list[tuple[int, float]]]:
    '''
    loads list of relevance queries, returns a dictionary that maps documents
    to a tuple with the docid and their relevance score 
    '''
    queries_to_judgements = defaultdict(list)
    with open(file) as f:
        reader = csv.reader(f)
        doc = next(reader)
        for doc in reader: 
            docid = int(doc[2])
            relevance = int(doc[-1])
            query = doc[0]
            queries_to_judgements[query].append((docid, relevance))

    return queries_to_judgements