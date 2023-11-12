import os 
from bs4 import BeautifulSoup
import requests
import json 
import re


class Scrapper():
  def __init__(self, base: str, limit: int = None, cache_file: str = None) -> None:
    '''
    Class to scrape search results from Google

    Also caches results to a json file
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
      self.cached_queries = list(cache.keys())


  def prepare_query(self, query: str) -> str:
    return query.replace(' ', '+')
  

  def prepare_url(self, query: str) -> str:
    url = f"{self.base}q={query}"
    if self.limit != None: 
      url += f'&num{self.limit}'
    return url 


  def add_to_cache(self, query: str, raw_text: str) -> None:
    self.cache[query] = raw_text
    with open(self.cache_file, 'w') as f: 
      json.dump(self.cache, f)


  def query(self, query: str):
    query = self.prepare_query(query) 
    
    if query in self.cache: 
      raw_text = self.cache[query]
    else: 
      url = self.prepare_url(query)
      raw_text = requests.get(url).text
      self.add_to_cache(query, raw_text)
    
    processed_response = self.process_response(raw_text)
    return processed_response
    

  def process_response(self, raw_text: str):
    soup = BeautifulSoup(raw_text, "html.parser")

    links = soup.find_all("a")
    headings = soup.find_all("h3")

    for link in links:
      for info in headings:
        get_title = info.getText()
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href and not 'accounts.google' in link_href and not 'support.google' in link_href:
          result_url = link.get('href').split("?q=")[1].split("&sa=U")[0]
          result_content = self.get_search_result_content(result_url)
          text = self.process_search_result_content(result_content)
          print(get_title)
          print(link)
          print(text)
          print("---------")


  def get_search_result_content(self, url: str) -> str | None:
    response = requests.get(url)
    if response != None: 
      return response.text 
    return None

  def process_search_result_content(self, content: str) -> list[str]:
    response_text = []
    soup = BeautifulSoup(content, 'html.parser')
    paragraphs = soup.find_all('p')
    [response_text.append(paragraph) for paragraph in paragraphs]
    return response_text

def main():
  query = "apple pie recipe"
  base = 'https://google.com/search?'
  scrapper = Scrapper(base, limit = 20, cache_file = 'files/cache.json')
  scrapper.query(query) 


if __name__ == '__main__':
  main()