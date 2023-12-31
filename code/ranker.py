
from collections import Counter
from indexing import PositionalInvertedIndex
from nltk.tokenize import TweetTokenizer
from sentence_transformers import CrossEncoder
import numpy as np 

"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
NOTE:
TODO's for hw2 are marked with `hw2` in the comments. See README.md for more details.
"""


class Ranker:
    '''
    The ranker class is responsible for generating a list of documents for a given query, ordered by their
    scores using a particular relevance function (e.g., BM25). A Ranker can be configured with any RelevanceScorer.
    '''

    # NOTE: (hw2) Note that `stopwords: set[str]` is a new parameter that you will need to use in your code.
    def __init__(self, index, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:
        '''
        Initializes the state of the Ranker object 
        '''
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords


    # NOTE: (hw2): `query(self, query: str) -> list[dict]` is a new function that you will need to implement.
    #            see more in README.md.
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Searches the collection for relevant documents to the query and returns a list 
        of documents ordered by their relevance (most relevant first).

        Args:
            query (str): The query to search for

        Returns:
            list: a list of dictionary objects with keys "docid" and "score" where docid is a
                  particular document in the collection and score is that document's relevance
        '''
        scores = []
        query_parts = self.tokenize(query)
        candidate_docs = {}

        #query_words_counts = Counter(query_parts)

        for i, part in enumerate(query_parts):
            doc_count = {}
            if part.lower() in self.stopwords:
                query_parts[i] = None
                continue
            postings = self.index.get_postings(part)
            if postings == None:
                continue
            for docid, count, _ in postings: 
                doc_count[part] = count
                candidate_docs[docid] = candidate_docs.get(docid, {}) | doc_count

        for docid in candidate_docs:
            counts = candidate_docs[docid]
            score = self.scorer.score(docid = docid, doc_word_counts = counts, query_parts = query_parts)
            scores.append((docid, score))

        return sorted(scores, key = lambda s: s[1], reverse = True)
        
    
class RelevanceScorer:
    def __init__(self, index, parameters) -> None:
        raise NotImplementedError


    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        raise NotImplementedError



class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index, parameters = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        query_word_counts = Counter(query_parts)

        scores = []
        for part in set(query_parts):
            if part in list(doc_word_counts.keys()):
                scores.append(doc_word_counts[part] * query_word_counts[part])
    
        if len(scores) == 0:
            return None 
        score = np.sum(scores)
        if score == 0:
            return None
        return score

class BM25(RelevanceScorer):
    def __init__(self, index, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        doc_freq_counts = Counter()
        query_word_counts = Counter(query_parts)

        for part in set(query_parts):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        doc_metadata = self.index.get_doc_metadata(docid)
        if len(doc_metadata) == 0:
            d = 0
        else:
            d = self.index.get_doc_metadata(docid)['num_tokens']
        # get parameters
        b = self.b
        k1 = self.k1
        k3 = self.k3

        scores = []
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())
        for k in mutual_terms:
            query_count = query_word_counts[k]
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]
            idf_ish_term = np.log((self.N - doc_freq + 0.5) / (doc_freq + 0.5))
            tf_ish_term = ((k1 + 1) * doc_count) / ((k1 * (1 - b + (b *(d/self.avg_d)))) + doc_count)
            qtf_ish_term = ((k3 + 1) * query_count) / (k3 + query_count)
            score = idf_ish_term * tf_ish_term * qtf_ish_term 
            scores.append(score)
        if np.sum(scores) == 0:
            return 0
        return np.sum(scores)


class PivotedNormalization(RelevanceScorer):
    def __init__(self, index, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self.N = index.get_statistics()['number_of_documents']
        self.avg_d = index.get_statistics()['mean_document_length']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        doc_freq_counts = Counter()
        query_word_counts = Counter(query_parts)

        for part in set(query_parts):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        doc_metadata = self.index.get_doc_metadata(docid)
        if len(doc_metadata) == 0:
            d = 0
        else:
            d = self.index.get_doc_metadata(docid)['total_token_count']
        b = self.b

        scores = []
        mutual_terms = set(doc_word_counts.keys()) & set(query_word_counts.keys())

        for k in mutual_terms:
            query_count = query_word_counts[k]
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]

            tf_ish_term = 1 + np.log(1 + np.log(doc_count))
            length_norm_term = 1 - b + (b * (d / self.avg_d))
            idf_ish_term = np.log((self.N + 1) / doc_freq)

            score = query_count * (tf_ish_term / length_norm_term) * idf_ish_term

            scores.append(score)

        score = np.sum(scores)
        return score




class TF_IDF(RelevanceScorer):
    def __init__(self, index, parameters={}) -> None:
        self.index = index
        self.parameters = parameters
        self.N = index.get_statistics()['number_of_documents']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        doc_freq_counts = Counter()

        for part in set(query_parts):
            term_metadata = self.index.get_term_metadata(part)
            if len(list(term_metadata.keys())) == 0:
                continue 
            doc_freq_counts[part] = term_metadata['n_docs']

        mutual_terms = set(doc_word_counts.keys()) & set(query_parts)

        doc_tf_idf = np.zeros(len(mutual_terms))
        for i, k in enumerate(mutual_terms):
            doc_count = doc_word_counts[k]
            doc_freq = doc_freq_counts[k]

            tf_term = np.log(doc_count + 1)
            idf_term = 1 + np.log(self.N /doc_freq)
            doc_tf_idf[i] = (tf_term * idf_term)
            
        if len(doc_tf_idf) == 0:
            return 0
        score = np.sum(doc_tf_idf)
        return score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str], cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        self.raw_text_dict = raw_text_dict 
        self.encoder = CrossEncoder(cross_encoder_model_name, max_length = 512)


    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        text = self.raw_text_dict.get(docid, '')
        if len(text) == 0 or len(query) == 0:
            return 0
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        score = self.encoder.predict((query, text))
        return score


