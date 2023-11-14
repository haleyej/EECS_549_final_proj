import os 
from bs4 import BeautifulSoup
import requests
import json 
import csv

class Scrapper():
  def __init__(self, base: str, limit: int = None, cache_file: str = None) -> None:
    '''
    Class to scrape search results from Google

    Caches results to a json file to prevent duplicate requests 
    '''
    self.base = base
    self.limit = limit

    # set up cache 
    if cache_file != None:
      if os.path.isfile(cache_file):
        with open(cache_file) as f: 
          cache = json.loads(f.read())
      else:
        # create file 
        open(cache_file, 'a').close()
        cache = {}
      self.cache_file = cache_file
      self.cache = cache
      self.cached_urls = list(cache.keys())


  def prepare_query(self, query: str) -> str:
    '''
    Prepares query 

    Returns a string where the white spaces have
    been converted to '+'

    e.g. 'apple pie recipe' -> 'apple+pie+recipe'

    ARGS:
      query: what's beign searched for 
    '''
    return query.replace(' ', '+')
  

  def prepare_url(self, query: str) -> str:
    '''
    Prepares URL for request 

    Returns a url made up of the base site, 
    query, and desired number of results

    ARGS:
      query: what's being looked up, already in a search friendly format
    '''
    url = f"{self.base}q={query}"
    if self.limit != None: 
      url += f'&num{self.limit}'
    return url 


  def add_to_cache(self, url: str, raw_text: str) -> None:
    '''
    Add a new search result to cache 

    Returns None

    ARGS:
      url: the website that was searched for 
      raw_text: unprocessed text of query 
    '''
    self.cache[url] = raw_text
    with open(self.cache_file, 'w') as f: 
      json.dump(self.cache, f)


  def query(self, query: str, save_path: str = None) -> dict[str]:
    '''
    Takes in a query, prepares a URL for search 

    If a query has already been searched for, it 
    will grab it from the cache instead 

    Will go through the full list of search results 
    and get the content of each page in the search results

    TO DO WHAT DOES THIS RETURN

    ARGS:
      query: what's being searched for, in natural language format
      save_path: file to save results to, defaults to None
    '''
    original_query = query[:]
    query = self.prepare_query(query) 
    url = self.prepare_url(query)

    if url in self.cached_urls:
      raw_text = self.cache[url]
    else: 
      raw_text = requests.get(url).text
      self.add_to_cache(query, raw_text)
    
    search_results = self.process_response(raw_text)

    if save_path != None:
      self.save_search_content(save_path, original_query, search_results)

    return search_results
    

  def process_response(self, raw_text: str) -> dict[str]:
    '''
    Processes search engine results 

    Grabs the content of each page in the search results

    ARGS:
      raw_text: raw text of search engine results 
    '''
    soup = BeautifulSoup(raw_text, "html.parser")

    results_to_content = {}
    links = soup.find_all("a")
    for link in links[25:40]:
      link_href = link.get('href')
      if "url?q=" in link_href and not "webcache" in link_href and not 'accounts.google' in link_href and not 'support.google' in link_href and not 'www.reddit.come/r/AskReddit' in link_href:
        result_url = link.get('href').split("?q=")[1].split("&sa=U")[0]
        result_content = self.get_search_result_content(result_url)
        page_content = self.process_search_result_content(result_content)
        results_to_content[result_url] = page_content
    
    return results_to_content


  def get_search_result_content(self, url: str) -> str | None:
    '''
    Gets the content of a google search result

    Returns the response text or none if the result 
    cannot be retrieved  

    ARGS: 
      url = url of google search result 
    '''
    if url in self.cached_urls:
      return self.cache[url]
    
    response = requests.get(url)
    if response != None: 
      self.add_to_cache(url, response.text)
      return response.text 
    else:
      return None


  def process_search_result_content(self, content: str) -> str:
    '''
    Process the response text from a google search result 

    Returns a string that roughly reconstructs the content 
    of the page
    
    Only collects headings & paragraphs for simplicity sake

    ARGS:
      content = response text from search result 
    '''
    soup = BeautifulSoup(content, 'html.parser')

    page_content = []
    headings_blocked_text = ['Sorry, you have been blocked', 'Why have I been blocked?', 'What can I do to resolve this?']

    for h in ['h1', 'h2', 'h3']:
      # get all paragraphs of a certain level 
      headings = soup.find_all(h)
      if headings == None: 
        continue
      for heading in headings: 
        # get heading text 
        heading_text = heading.get_text()
        if heading_text in headings_blocked_text or 'You are unable to access' in heading_text: 
          continue
        heading_text = heading_text.strip()
        page_content.append(heading_text)
        # get all content after heading
        for elem in heading.next_siblings: 
          if elem.name and elem.name.startswith('h'):
            break
          elif elem.name == 'p':
            paragraph_text = elem.get_text()
            paragraph_text = paragraph_text.strip()
            if len(paragraph_text) > 0:
              page_content.append(paragraph_text)

    if len(page_content) > 0:
      return " ".join(page_content)
    else:
      return None

  def save_search_content(self, save_path: str, query: str, search_results: dict[str]) -> None:
    '''
    Saves processed search results to csv 

    Does not return anything 

    ARGS:
      save_path: name of file to save results to 
    '''
    with open(save_path, 'a') as f: 
      writer = csv.writer(f)
      for url, page_content in list(search_results.items()):
        if page_content == None:
          continue
        writer.writerow([query, url, page_content])


def main():
  query = "apple pie recipe"
  base = 'https://google.com/search?'
  scrapper = Scrapper(base, limit = 20, cache_file = 'files/cache.json')
  scrapper.query(query, save_path = 'files/processed_search_results/results.csv') 


if __name__ == '__main__':
  main()