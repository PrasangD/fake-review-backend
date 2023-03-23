import webScrapping as ws
import trainer as tr
import pandas as pd
import predict as p
links = []
names = []
reviews = []
df = {}
df_list = []
def predict(url):
    data_str = ""
    print("URLS")
    print(url)
    soup = ws.html_code(url)
    reviews = []
    for l in soup.find_all('a', class_= 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'):
            x = 'https://www.amazon.in' + l.get('href')
            links.append(x)
    # Get names of product # Get cust reviews
    for l in range(0,5):
        # len(links)
        s = ws.html_code(links[l])
        for i in s.find_all('span', class_ = "a-size-large product-title-word-break"):
            names.append(i.get_text())
        
        for j in s.find_all("div", class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content"):
            data_str = data_str+j.get_text()
            reviews = data_str.split("\n")
        df = {"REVIEW_TEXT":reviews}
        df_obj = pd.DataFrame.from_dict(df)
        df_obj_ret = tr.preprocessPandas(df_obj)
        df_dict = p.predict(df_obj_ret,names[l])
        df_list.append(df_dict)
    print(df_list)
    return df_list