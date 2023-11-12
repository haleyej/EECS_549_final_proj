import os 
from bs4 import BeautifulSoup
import requests
from requests import Response


class Scrapper():
  def __init__(self, base: str, limit: int = None) -> None:
    self.base = base
    self.limit = limit

  def prepare_query(self, query: str) -> str:
    return query.replace(' ', '+')
  
  def prepare_url(self, query: str) -> str:
    url = f"{self.base}q={query}"
    if self.limit != None: 
      url += f'&num{self.limit}'
    return url 

  def query(self, query: str):
    query = self.prepare_query(query) 
    url = self.prepare_url(query)
    print(url)
    raw = requests.get(url)
    print(raw)
    

  def process_response(self, raw: Response):
    soup = BeautifulSoup(raw.text, "html.parser")
    links = soup.find_all("a")
    headings = soup.find_all("h3")

    for link in links:
      for info in headings:
        get_title = info.getText()
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
          print(get_title)
          print(link.get('href').split("?q=")[1].split("&sa=U")[0])
          print("------")
      
  def save_results(self, outfile: str):
    pass
      

def main():
  query = "StackOverflow"
  base = 'https://google.com/search?'
  scrapper = Scrapper(base, limit = 10)
  scrapper.query(query) 


if __name__ == '__main__':
  main()