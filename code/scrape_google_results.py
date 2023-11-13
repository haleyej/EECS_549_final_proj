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

def run_scrapper(base: str , queries: str, cache_file: str, save_path: str):
    scrapper = Scrapper(base, cache_file = cache_file)

    for query in queries: 
        _ = scrapper.query(query, save_path = save_path)
        print(_)

def main():
    base = 'https://google.com/search?'
    queries = load_queries("eval/evaluation_queries.csv")
    run_scrapper(base, queries[:5], cache_file = 'files/cache.json', save_path = 'files/processed_search_results/search.csv')

if __name__ == '__main__':
    main()