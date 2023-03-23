from datasets import Dataset
import pandas as pd
import warnings
import re
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Lemmatize the text
lemmer = WordNetLemmatizer()
stop_words = stopwords.words('english')

warnings.filterwarnings('ignore')

def readCsv(dataset):
    return pd.read_csv(dataset)


def convertToPandas(dataset):
    return pd.DataFrame.from_dict(dataset)

def convertToDataset(dataset):
    return Dataset.from_pandas(dataset)


def tokenize(dataset):
    return dataset


def preprocessPandas(df):
    df["length"] = df['REVIEW_TEXT'].apply(len)
    stop_words = set(stopwords.words('english'))
    # df['no_of_stopwords'] = df['REVIEW_TEXT'].str.split().apply(lambda x: len(set(x) & stop_words)) 
    df['REVIEW_TEXT'] = df['REVIEW_TEXT'].replace(to_replace='\s+',value=' ',regex=True)
    length = []
    for review in df['REVIEW_TEXT']:
        length.append(len(set(review)))

    length_df = pd.DataFrame(length)
    # df["Vocabulary"] = length_df
    # Remove HTTP tags
    df['review_processed'] = df['REVIEW_TEXT'].map(lambda x : ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
    df['review_processed'] = df['review_processed'].map(lambda x: x.lower())
    df['review_processed'] = df['review_processed'].map(lambda x: re.sub(r'[^\w\s]', '', x))
    df['review_processed'] = df['review_processed'].map(lambda x : re.sub(r'[^\x00-\x7F]+',' ', x))
    df['review_processed'] = df['review_processed'].map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split() if w not in stop_words]))
    
    return df
