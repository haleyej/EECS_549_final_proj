import xgboost as xgb
import numpy as np 
from tqdm import tqdm 
from ranker import Ranker, BM25
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from relevance import map_score, ndcg_score



class XGBRankerFeatures():
    '''
    handy little wrapper class for xgboost
    so we can do cross val easily 
    '''
    def __init__(self, post_index, 
                 comment_index, 
                 bm25_params: dict[str, dict[str, int]],
                 cross_encoder_scores: dict[int, float],
                 karma_scores: dict[int, tuple[int]],
                 sentiment_scores: dict[int, float])  -> None:

        # features
        self.karma_scores = karma_scores
        self.cross_encoder_score = cross_encoder_scores
        self.sentiment_scores = sentiment_scores
        
        #BM25
        post_parmas = bm25_params.get('post')
        comment_params = bm25_params.get('comment')
        self.post_BM25 = BM25(post_index, post_parmas)
        self.comment_BM25 = BM25(comment_index, comment_params)

        #index
        self.post_index = post_index
        self.comment_index = comment_index


    def get_karma_score(self, docid) -> tuple[int]:
        return self.karma_scores.get(docid, (0, 0))
    

    def get_sentiment_score(self, docid) -> float:
        return self.sentiment_scores.get(docid, 0)


    def get_bm25(self, docid:int, doc_word_counts: dict[str, int], query_word_parts: list[str], index_type: str = 'post'):
        if index_type == 'post':
            score = self.post_BM25.score(docid, doc_word_counts, query_word_parts)
        elif index_type == 'comment':
            score = self.comment_BM25.score(docid, doc_word_counts, query_word_parts)
        return score
    

    def get_cross_encoder_score(self, docid: str):
        return self.cross_encoder_score.get(docid, 0)


    @staticmethod
    def get_doc_word_counts(index, docid: str) -> dict[str, int]:
        doc_word_counts = {}
        words = list(index.term_metadata.keys())
        for word in words:
            postings = index.get_postings(word)
            for posting in postings:
                posting_docid = posting[0]
                if posting_docid == docid:
                    count = posting[1]
                    doc_word_counts[word] = count

        return doc_word_counts
    
    def get_ranker_features(self, X: list[int], query_word_parts: list[str]) -> list[list]:
        doc_features = []
        for docid in X: 
            post_doc_word_counts = self.get_doc_word_counts(self.post_index, docid)
            post_bm25 = self.get_bm25(docid, post_doc_word_counts, query_word_parts, index_type = 'post')

            comment_doc_word_counts = self.get_doc_word_counts(self.comment_index, docid)
            comment_bm25 = self.get_bm25(docid, comment_doc_word_counts, query_word_parts, index_type = 'comment')

            post_karma, comment_karma = self.get_karma_score(docid)

            post_len = len(list(post_doc_word_counts.keys()))
            comment_len = len(list(comment_doc_word_counts.keys()))

            sentiment = self.get_sentiment_score(docid)

            cross_encoder_score = self.get_cross_encoder_score(docid)

            feature_vec = [post_bm25, comment_bm25, post_karma, comment_karma, post_len, comment_len, sentiment, cross_encoder_score]
            if len(X) == 1:
                return feature_vec
            doc_features.append(feature_vec)
        
        return doc_features


class XGBRankerWrapper():
    def __init__(self, feature_preparer, stopwords:list[str], 
                 doc_preprocessor, ranker, 
                 objective:str = 'rank:ndcg', learning_rate:int = 0.1,
                 gamma:int = 0.5, max_depth:int = 10, n_estimators:int = 100, 
                 tree_method:str = 'hist', lambdarank_pair_method:str = 'topk', 
                 lambdarank_num_pair_per_sample: int = 8) -> None:
        
        # ir infastructure
        self.doc_preprocessor = doc_preprocessor
        self.feature_preparer = feature_preparer
        self.stopwords = stopwords
        self.ranker = ranker

        # hyperparameters
        self.objective = objective
        self.learning_rate = learning_rate 
        self.gamma = gamma 
        self.max_depth = max_depth 
        self.n_estimators = n_estimators
        self.tree_method = tree_method
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample



    def fit(self, query_to_relevance: dict) -> None:
        ranker = xgb.XGBRanker(tree_method = self.tree_method,
                               lambdarank_num_pair_per_sample = self.lambdarank_num_pair_per_sample, 
                               lambdarank_pair_method = self.lambdarank_pair_method,
                               objective = self.objective, 
                               learning_rate = self.learning_rate, 
                               gamma = self.gamma, 
                               max_depth = self.max_depth)
    

        qids = []
        X = []
        y = []
        for i, query in tqdm(enumerate(list(query_to_relevance.keys()))):
            ratings = query_to_relevance[query]
            query_word_parts = self.tokenize_query(query)

            for docid, rel in ratings: 
                y.append(rel)
                feature_vec = self.feature_preparer.get_ranker_features([docid], query_word_parts)
                X.append(feature_vec)
                qids.append(i)

        X = np.array(X)
        y = np.array(y)
        qids = np.array(qids)

        print(X.shape, y.shape, qids.shape)
        ranker.fit(X, y, qid = qids)

        self.model = ranker
    

    def tokenize_query(self, query: str) -> list[str]:
        query = query.lower()
        parts =  self.doc_preprocessor.tokenize(query)
        return [part for part in parts if part not in self.stopwords]


    def get_params(self) -> dict:
        if hasattr(self, 'model'):
            return self.model.get_params
        return {}


    def query(self, query: str, cutoff: int = 100):
        query_word_parts = self.tokenize_query(query)
        base = self.ranker.query(query)

        reranked_docs = []
        for docid, _ in base[:cutoff]:
            features = np.array(self.feature_preparer.get_ranker_features([docid], query_word_parts)).reshape(1, -1)
            prediction = self.model.predict(features)[0]
            reranked_docs.append(docid, prediction)


        reranked_docs = sorted(reranked_docs, key = lambda s: s[1], reverse = True)
        search_results = reranked_docs.extend(base[cutoff:])

        return search_results


def main():
    from sklearn.datasets import make_classification
    seed = 1994
    X, y = make_classification(random_state=seed)
    rng = np.random.default_rng(seed)
    n_query_groups = 3
    qid = rng.integers(0, 3, size=X.shape[0])

    # Sort the inputs based on query index
    sorted_idx = np.argsort(qid)
    X = X[sorted_idx, :]
    y = y[sorted_idx]
    qid = qid[sorted_idx]

    ranker = xgb.XGBRanker(tree_method="hist", lambdarank_num_pair_per_sample=8, objective="rank:ndcg", lambdarank_pair_method="topk")
    ranker.fit(X, y, qid=qid)


if __name__ == "__main__":
    main()