'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
Use libraries such as tqdm, orjson, collections.Counter, shelve if you need them.
DO NOT use the pickle module.
NOTE: 
There are a few changes to the indexing file for HW2.
The changes are marked with a comment `# NOTE: changes in this method for HW2`. 
Please see more in the README.md.
'''
from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from nltk.tokenize import TweetTokenizer

import gzip
import shutil
import numpy as np


class IndexType(Enum):
    # the three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    # NOTE: You don't need to support these other three
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class PositionalInvertedIndex():
    def __init__(self) -> None:
        self.statistics = defaultdict(Counter)  
        self.statistics = {'vocab': {}} 
        self.index = {}  
        self.vocabulary = set()  
        self.document_metadata = {}
        self.term_metadata = {}
        self.index_segment = 0

    def remove_doc(self, docid:int) -> None:
        for token, docs in list(self.index.items()):
            for i, doc in enumerate(docs):
                if int(doc[0]) == int(docid):
                    del docs[i]
                    break
            if len(docs) == 0:
                del self.index[token]
                self.vocabulary.remove(token)
                del self.term_metadata[token]

        del self.document_metadata[docid]

            
    def add_doc(self, docid:int, tokens:list[str]) -> None:
        counts = Counter(tokens)
        for token in set(tokens):
            if type(token) == type(None):
                continue
            else:
                self.vocabulary.add(token)
            term_count = counts[token]
            indices = np.where(np.array(tokens) == token)[0]
            indices = list([int(i) for i in indices])
            if token in self.index.keys():
                if self.index[token] == None:
                    continue
                new = (docid, term_count, indices)
                current = self.index[token]
                current.append(new)
                self.index[token] = current
                self.term_metadata[token]['count']  = self.term_metadata[token].get('count', 0) + term_count 
                self.term_metadata[token]['n_docs'] = self.term_metadata[token].get('n_docs', 0) + 1
            else:
                self.index[token] = [(docid, term_count, indices)]
                self.term_metadata[token] = {}
                self.term_metadata[token]['count']  = term_count 
                self.term_metadata[token]['n_docs'] =  1
        num_tokens = len(tokens)
        unique_tokens = set(tokens)
        unique_tokens.discard(None)


        self.document_metadata[docid] = {'num_tokens': num_tokens, 'num_unique_tokens': len(unique_tokens)}

    def get_postings(self, term: str) -> list:
        return self.index.get(term, None)

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        return self.term_metadata.get(term, {})

    def get_statistics(self) -> dict[str, int]:
        doc_metadata =  self.document_metadata

        # calculate stats 
        tokens_per_doc = [doc_metadata[k]['num_tokens'] for k in list(doc_metadata.keys())]
        all_tokens = [doc_metadata[k]['num_tokens'] for k in list(doc_metadata.keys())]
        stats = {
                    'number_of_documents': len(list(doc_metadata.keys())), 
                    'mean_document_length': np.nan_to_num(np.mean(tokens_per_doc)), 
                    'total_token_count': np.sum(all_tokens),
                    'unique_token_count': len(self.vocabulary) 
                }

        return stats

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod 
    def load_docs(doc: dict, text_key: str, document_preprocessor, doc_augment_dict: dict[int, list[str]] | None = None):
        text = doc[text_key]
        docid = int(doc['docid'])
        if doc_augment_dict != None:
            augmentations = doc_augment_dict.get(docid)
            if augmentations != None: 
                for augmentation in augmentations: 
                    text += (" " + augmentation)
        tokens = document_preprocessor.tokenize(text)
        return (docid, tokens)

    @staticmethod 
    def add_to_index(doc_info:list, do_not_index:list[str], index):
        docid, tokens = doc_info
        for i, token in tqdm(enumerate(tokens)):
            if token.lower() in do_not_index:
                tokens[i] = None
        index.add_doc(docid, tokens)
        return index

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str, 
                     document_preprocessor, stopwords: set[str], 
                     minimum_word_frequency: int, text_key="text",
                     max_docs=-1, doc_augment_dict: dict[int, list[str]] | None = None):

        index = PositionalInvertedIndex()
        
        i = 0
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, 'r') as f:
                doc = f.readline()
                while doc and (i < max_docs or max_docs == -1):
                    doc = json.loads(doc)
                    doc_info = __class__.load_docs(doc, text_key, document_preprocessor, doc_augment_dict)
                    index = __class__.add_to_index(doc_info, stopwords, index)
                    doc = f.readline() 
                    i += 1

        if minimum_word_frequency > 1:
            metadata = index.term_metadata
            for term in tqdm(list(metadata.keys())):
                if metadata[term]['count'] < minimum_word_frequency:
                    postings = index.get_postings(term)
                    for posting in postings:
                        docid, _ = posting
                        index.document_metadata[docid]['num_unique_tokens'] = index.document_metadata[docid]['num_unique_tokens'] - 1
                    index.vocabulary.remove(term)
                    del index.index[term]
                    del index.term_metadata[term]
        return index



def main():
    tokenizer = TweetTokenizer()
    idx = Indexer().create_index('InvertedIndex', "sample_docs.jsonl", tokenizer, ['the'], 5)

if __name__ == '__main__':
     main()