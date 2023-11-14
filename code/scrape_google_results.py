from scrapper import Scrapper
import csv 


def load_queries(queries_path: str) -> list[str]:
    '''
    file_path is a csv 
    '''
    with open(queries_path, 'r') as f: 
        reader = csv.reader(f)
        queries = [line[0] for line in reader if line != ['Query']]

    return queries

def run_scrapper(base: str, limit: int, queries: str, cache_file: str, save_path: str) -> None:
    scrapper = Scrapper(base, limit = limit, cache_file = cache_file)

    for query in queries: 
        _ = scrapper.query(query, save_path = save_path)

def main():
    base = 'https://google.com/search?'
    queries = load_queries("eval/evaluation_queries.csv")
    run_scrapper(base, 20, queries[20:25], cache_file = 'files/cache.json', save_path = 'files/processed_search_results/results.csv')

if __name__ == '__main__':
    main()