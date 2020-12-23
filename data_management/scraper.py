import requests
from bs4 import BeautifulSoup

URL = 'https://www.instagram.com/animit_kulkarni/'
page = requests.get(URL)

soup = BeautifulSoup(page.content, 'html.parser')

print(soup)