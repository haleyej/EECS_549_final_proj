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
from document_preprocessor import Tokenizer, RegexTokenizer
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


class InvertedIndex:
    '''
    The base interface representing the data structure for all index classes.
    The functions are meant to be implemented in the actual index classes and not as part of this interface.
    '''

    def __init__(self) -> None:
        self.statistics = defaultdict(Counter)  # the central statistics of the index
        self.statistics = {'vocab': {}}  # the central statistics of the index
        self.index = {}  # the index
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        # term metadata 
        self.term_metadata = {}
        # OPTIONAL if using SPIMI, use this variable to keep track of the index segments.
        self.index_segment = 0


    def remove_doc(self, docid: int) -> None:
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        raise NotImplementedError

    def get_postings(self, term: str) -> list[tuple[int, str]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        raise NotImplementedError

    # NOTE: changes in this method for HW2
    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        raise NotImplementedError

    # NOTE: changes in this method for HW2
    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    '''
    An inverted index implementation where everything is kept in memory.
    '''

    # NOTE: changes in this class for HW2
    def __init__(self) -> None:
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        # keep docs associated with first 500 words
        self.docs_to_words = {}

    def remove_doc(self, docid: int) -> None:
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

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        '''
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length, etc.)

        Arguments:
            docid [int]: the identifier of the document

            tokens list[str]: the tokens of the document. Tokens that should not be indexed will have 
            been replaced with None in this list. The length of the list should be equal to the number
            of tokens prior to any token removal.
        '''
        self.docs_to_words[docid] = tokens[:500]
        counts = Counter(tokens)
        for token in set(tokens):
            if token == None:
                continue
            self.vocabulary.add(token)
            term_count = counts[token]
            if token in self.term_metadata.keys():
                if self.index[token] == None:
                    continue
                new = (docid, term_count)
                current = self.index[token]
                # idx = bisect.bisect_left(current, new)
                # current.insert(idx, new)
                current.append(new)
                self.index[token] = current

                self.term_metadata[token]['count']  = self.term_metadata[token].get('count', 0) + term_count 
                self.term_metadata[token]['n_docs'] = self.term_metadata[token].get('n_docs', 0) + 1
            else:
                self.index[token] = [(docid, term_count)]
                self.term_metadata[token] = {}
                self.term_metadata[token]['count']  = term_count 
                self.term_metadata[token]['n_docs'] =  1

        # doc metadata
        num_tokens = len(tokens)
        unique_tokens = set(tokens)
        unique_tokens.discard(None)

        self.document_metadata[docid] = {'num_unique_tokens': len(unique_tokens), 'total_token_count': num_tokens}

        length = len(tokens)
        if length == 0:
            self.document_metadata[docid] = {'length': 0, 'unique_tokens': 0}
            return
         
    def get_postings(self, term: str) -> list[tuple[int, str]]:
        '''
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.
        
        Arguments:
            term [str]: the term to be searched for

        Returns:
            list[tuple[int,str]] : A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document.
        '''
        return self.index.get(term, None)

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        '''
        For the given document id, returns a dictionary with metadata about that document. Metadata
        should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)             
        '''
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str) -> dict[str, int]:
        '''
        For the given term, returns a dictionary with metadata about that term in the index. Metadata
        should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole.          
        '''        
        return self.term_metadata.get(term, {})

    def get_statistics(self) -> dict[str, int]:
        '''
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:

            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)                
        '''
        doc_metadata = self.document_metadata

        tokens_per_doc = [doc_metadata[k].get('total_token_count', 0) for k in list(doc_metadata.keys())]
        all_tokens = [doc_metadata[k].get('total_token_count', ) for k in list(doc_metadata.keys())]
        stats = {
                    'number_of_documents': len(list(doc_metadata.keys())), 
                    'mean_document_length': np.nan_to_num(np.mean(tokens_per_doc)), 
                    'total_token_count': np.sum(all_tokens),
                    'unique_token_count': len(self.vocabulary) 
                }
        return stats

    def save(self, index_directory_name) -> None:
        '''
        Saves the state of this index to the provided directory. The save state should include the
        inverted index as well as any meta data need to load this index back from disk
        '''
        if os.path.exists(index_directory_name):
            shutil.rmtree(index_directory_name)
        os.mkdir(index_directory_name)
        with open(os.path.join(index_directory_name, 'BasicIndex' + ".json"), "w") as f:
            f.write(json.dumps(self.index))

    def load(self, index_directory_name) -> None:
        '''
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save()
        '''
        if os.path.exists(os.path.join(index_directory_name, 'BasicIndex' + ".json")):
            with open(os.path.join(index_directory_name, 'BasicIndex' + ".json"), "r") as f:
                self.index = json.loads(f.read())

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod 
    def load_docs(doc: dict, text_key: str, document_preprocessor: Tokenizer, doc_augment_dict: dict[int, list[str]] | None = None):
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
    def add_to_index(doc_info:list, do_not_index:list[str], index:InvertedIndex):
        docid, tokens = doc_info
        for i, token in tqdm(enumerate(tokens)):
            if token.lower() in do_not_index:
                tokens[i] = None
        index.add_doc(docid, tokens)
        return index

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str, 
                     document_preprocessor: Tokenizer, stopwords: set[str], 
                     minimum_word_frequency: int, text_key="text",
                     max_docs=-1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        The Index class' static function which is responsible for creating an inverted index

        Parameters:        

        index_type [IndexType]: This parameter tells you which type of index to create, e.g., a BasicInvertedIndex.

        dataset_path [str]: This is the file path to your dataset

        document_preprocessor: This is a class which has a 'tokenize' function which would read each document's text and return back a list of valid tokens.

        stopwords [set[str]]: The set of stopwords to remove during preprocessing or `None` if no stopword preprocessing is to be done.

        minimum_word_frequency [int]: This is also an optional configuration which sets the minimum word frequency of a particular token to be indexed. If the token does not appear in the document atleast for the set frequency, it will not be indexed. Setting a value of 0 will completely ignore the parameter.

        text_key [str]: the key in the JSON to use for loading the text. 

        max_docs [int]: The maximum number of documents to index. Documents are processed in the order they are seen

        '''        
        index = BasicInvertedIndex()
        
        i = 0
        if dataset_path.endswith(".gz"):
            with gzip.open(dataset_path, 'r') as f:
                doc = f.readline()
                while doc and (i < max_docs or max_docs == -1):
                    doc = json.loads(doc)
                    doc_info = __class__.load_docs(doc, text_key, document_preprocessor, doc_augment_dict)
                    index = __class__.add_to_index(doc_info, stopwords, index)
                    doc = f.readline()
                    i += 1
        elif dataset_path.endswith(".jsonl"):
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
    tokenizer = RegexTokenizer(r"\w+")
    idx = Indexer().create_index('InvertedIndex', "sample_docs.jsonl", tokenizer, ['the'], 5)

if __name__ == '__main__':
     main()