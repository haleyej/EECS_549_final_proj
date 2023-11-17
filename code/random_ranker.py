import numpy as np 

class RandomRanker:
    '''
    Dummy ranker for baseline comparison
    '''
    def __init__(self, index, n_results: int = 10):
        self.index = index 
        self.n_results = n_results
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Completely ignores query 
        Returns random documents with random scores
        '''
        docs = self.index.document_metadata
        scores = []
        for i in range(self.n_results):
            doc = np.random.choice(list(docs.keys()))
            print(doc)
            score = np.random.rand()
            scores.append((doc, score))

        return sorted(scores, key = lambda s: s[1], reverse = True)