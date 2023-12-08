import xgboost as xgb
import numpy as np 
from ranker import Ranker
from sklearn.model_selection import GroupKFold
from skopt import BayesSearchCV
from relevance import map_score, ndcg_score

from nltk import TwitterTokenizer


class XGBRankerWrapper():
    '''
    handy little wrapper class for xgboost
    so we can do cross val easily 
    '''
    def __init__(self, karma_features: dict[int, tuple[int]], 
                 bm25_params: dict[str, dict[str, int]],
                 BM25, 
                 doc_preprocessor, 
                 objective:str = 'rank: ndcg', learning_rate:int = 0.1,
                 gamma:int = 0.5, max_depth:int = 10, n_estimators:int = 100, 
                 tree_method:str = 'hist', lambdarank_pair_method:str = 'topk', 
                 lambdarank_num_pair_per_sample: int = 8)  -> None:
        self.objective = objective
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_depth = max_depth
        self.n_estimators = n_estimators 
        self.tree_method = tree_method
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample

        self.karma_features = karma_features
        self.bm25_params = bm25_params
        self.BM25 = BM25
        self.tokenizer = doc_preprocessor


    def get_karma_features(self, docid) -> tuple[int]:
        post_karma, comment_karma = self.karma_features.get(docid, (0, 0))
        return (post_karma, comment_karma)


    def get_tuned_bm25(self, query_parts: str, index_type: str):
        params = self.bm25_params.get(index_type)

        b = params.get('b', 0)
        k1 = params.get('k1', 0)
        k3 = params.get('k3', 0)

        return (b, k1, k3)
    
    def get_ranker_features(X, y):
        pass 


    def fit(self, X, y, qid:list[int]) -> None:
        ranker = xgb.XGBRanker(tree_method = self.tree_method,
                               lambdarank_num_pair_per_sample = self.lambdarank_num_pair_per_sample, 
                               lambdarank_pair_method = self.lambdarank_pair_method,
                               objective = self.objective, 
                               learning_rate = self.learning_rate, 
                               gamma = self.gamma, 
                               max_depth = self.max_depth)
        
        ranker.fit(X, y, qid=qid)

        self.model = ranker
        return self
    
    def prepare_data(self, docs):
        for docid in docs: 
            post_karma, comment_karma = self.get_karma_features(docid)
            bm25 = self.get_


    def get_params(self, deep) -> dict:
        if hasattr(self, 'model'):
            return self.model.get_params
        return {}

    def predict(self, x):
        return self.model.predict(x)


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