import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import nltk
#import pandas_profiling as pp
from markupsafe import escape
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, cross_val_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# model building imports
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.layers import Conv1D, SimpleRNN, Bidirectional, MaxPooling1D, GlobalMaxPool1D, LSTM, GRU
from keras.models import Sequential
from keras.regularizers import L1L2
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import yake
import pickle


#%matplotlib inline

# matplotlib defaults
plt.style.use("ggplot")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

import warnings
warnings.filterwarnings('ignore')
nltk.download('omw-1.4')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



df = pd.read_json("data/News_Category_Dataset_v3.json", lines= True)
df = df[["headline", "category"]]
df = df.iloc[:500] # TRIAL

import yake
kw_extractor = yake.KeywordExtractor()
text = """spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""
language = "en"
max_ngram_size = 1
deduplication_threshold = 0.1
numOfKeywords = 3
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)
keywords = custom_kw_extractor.extract_keywords(text)
for kw in keywords:
    print(kw)


all_words = []

for i in df.index:
    string = df.loc[i]["headline"]
    string = string.lower()


    keywords = custom_kw_extractor.extract_keywords(string)



    for s in keywords:
        all_words.append(s[0])

all_words


test_df = pd.DataFrame(columns= all_words)

number = 0
for i in df.index:
    string = df.loc[i]["headline"]
    string = string.lower()

    keywords = custom_kw_extractor.extract_keywords(string)


    test_df.loc[number] = 0

    for s in keywords:
        test_df.loc[number, s[0]] = 1



    number += 1


test_df["catgory"] = df["category"]

mapping = {}
mapping2 = {}
for cat, num in enumerate(test_df["catgory"].unique()):
    mapping[cat] = num
    mapping2[num] = cat



test_df["catgory"] = test_df["catgory"].map(mapping2)


x = test_df.drop(columns = "catgory")
y = test_df["catgory"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

model = KMeans(n_clusters = 42, random_state= 0)
model.fit(x_train)

pickle.dump(model, open('c.sav','wb'))
pickle.dump(model, open('m.sav', 'wb'))



predictions = model.predict(x_test)

accuracy_score(y_test, predictions)
