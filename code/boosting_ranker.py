import xgboost as xgb
import numpy as np 
from ranker import Ranker
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV


class BoostingRanker():
    def __init__(self, params:dict) -> None:
        self.objective = params.get('objective', 'rank:ndcg')
        self.n_estimators = params.get('n_estimators', 500)
        

    def fit(self, X, y, qid:list[int], tuning:bool = False, param_grid:dict = None) -> None:
        if tuning:
            xgb_ranker = xgb.XGBRanker(tree_method = "hist", lambdarank_pair_method = 'topk')
            folds = StratifiedGroupKFold(shuffle = False)
            cv_search = GridSearchCV(xgb_ranker, param_grid, cv = folds)
            cv_search.fit(X, y, qid)
            
            self.model = cv_search.best_estimator_
        else:
            self.model = xgb.XGBRanker(tree_method = "hist", 
                                        n_estimators = self.n_estimators,
                                        lambdarank_num_pair_per_sample = 8, 
                                        objective = self.objective, 
                                        lambdarank_pair_method = "topk")
            self.model.fit(X, y, qid)
        

    def predict(self, query:str):
        return self.model.predict(query)


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

    ranker = BoostingRanker()
    ranker.fit(X, y, qid)
     

if __name__ == "__main__":
    main()