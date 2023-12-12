import numpy as np 

class RandomRanker:
    '''
    Dummy ranker for baseline comparison
    '''
    def __init__(self, index, n_results: int = 10):
        self.index = index 
        self.n_results = n_results
        self.docs = list(self.index.document_metadata.keys())
    
    def query(self, query: str) -> list[tuple[int, float]]:
        '''
        Completely ignores query 
        Returns random documents with random scores
        '''
        scores = []
        scored_docs = []
        for i in range(self.n_results):
            doc = np.random.choice(self.docs)
            if doc in scored_docs:
                continue
            score = np.random.rand()
            scores.append((doc, score))
            scored_docs.append(doc)

        return sorted(scores, key = lambda s: s[1], reverse = True)