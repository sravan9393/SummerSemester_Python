import requests
from bs4 import BeautifulSoup

website_url=requests.get('https://en.wikipedia.org/wiki/Deep_learning').text

soup=BeautifulSoup(website_url,'html.parser')
print("Title of web page:",soup.title)
for link in soup.find_all('a',{'class':'mw-redirect'}):
    print(link.get('href'))
