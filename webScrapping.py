import requests
from bs4 import BeautifulSoup
# from selectorlib import Extractor

cust_name = []
HEADERS = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/39.0.2171.95 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})


def getData(url):
    r = requests.get(url,headers=HEADERS)
    # print(r.text)
    return r.text


def html_code(url):
    htmlData = getData(url)
    soup = BeautifulSoup(htmlData,'html.parser')
    return soup

def webScrape_CustName(url):

    soup = html_code(url)
    data_str = ""
    for item in soup.find_all("span", class_="a-profile-name"):
        data_str = data_str+item.get_text()
        cust_name.append(data_str)
        data_str = ""

    return cust_name


def webScrape_Reviews(url):
    print(url)
    soup = html_code(url)
    # print(soup.prettify())
    data_str = ""
    # print(soup)
    for item in soup.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):
        data_str = data_str+item.get_text()
        # print(data_str)

    result = data_str.split("\n")
    # print(list(result))\
    # res = []
    # print(len(result))
    # for i in result:
    #     if i is None or i == ' ':
    #         continue
    #     else:
    #         res.append(i)

    return result


def getProductLinks(url):
    soup = html_code(url)
    links = []
    for l in soup.find_all('a', class_= 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'):
        x = 'https://www.amazon.in' + l.get('href')
        links.append(x)
    
    return links

def productName(url):
    soup = html_code(url)
    names = []

    for n in soup.find_all('span', class_ = "a-size-large product-title-word-break"):
        names.append(n.get_text())
    
    return names



# webScrape_Reviews('https://www.amazon.in/Apple-iPhone-13-128GB-Midnight/dp/B09G9HD6PD/ref=sr_1_3?crid=28I4I1AJ25YXT&keywords=iphone+13&qid=1655992731&s=electronics&sprefix=iphone+13%2Celectronics%2C697&sr=1-3')

    






    

  
