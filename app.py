import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import nltk


nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import re, string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud

app = Flask(__name__)


def jalan():
    df = pd.read_csv("Reviews1000.csv")
    # menampilkan text
    text = df["Text"]
    text = list(text)

    # Lower
    low_text = []
    for item in text:
        low_text.append(item.lower())

    # Remove punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    clean_text = []
    for item in low_text:
        clean_text.append(regex.sub('', item))

    # Import stopwords English
    stopword = stopwords.words('english')

    # Tokenize word
    token_text = []
    for item in clean_text:
        token_text.append(nltk.word_tokenize(item))

    # Remove stopwords english
    stopword_text = []
    for item in token_text:
        stopwording_text = [word for word in item if word not in stopword]
        stopword_text.append(" ".join(stopwording_text))

    # Remove digits
    nodigit_text = []
    for item in clean_text:
        nodigit_text.append(''.join(i for i in item if not i.isdigit()))

    regex = re.compile('[%s]' % re.escape(string.punctuation))

    def preProcess(textString):
        # function yg menerima 1 buah review berupa string
        textString = textString.lower()
        textString = regex.sub('', textString)
        textString = nltk.word_tokenize(textString)
        textString = [word for word in textString if word not in stopword]
        textString = " ".join(textString)
        textString = (''.join(i for i in textString if not i.isdigit()))

        return textString

    textClean = []
    for review in text:
        textClean.append(preProcess(review))

    vectorizer = CountVectorizer()  # memanggil class untuk method BoW
    X = vectorizer.fit_transform(textClean)  # mentransform data review ke BoW
    X.toarray()

    vectorizer.get_feature_names()

    #vectorizer.get_feature_names().shape  # jumlah model tabel baru

    # modelkan fitur berupa vektor BoW dengan sklearn untuk K-Means
    kmeans = KMeans(5, random_state=153).fit(X)  # nilai random state terserah tetapi harus konsisten dari awal-akhir


    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()

    # menyiapkan index dari tiap review sesuai cluster
    cluster0id = np.where(kmeans.labels_ == 0)[0].tolist()
    cluster1id = np.where(kmeans.labels_ == 1)[0].tolist()
    cluster2id = np.where(kmeans.labels_ == 2)[0].tolist()
    cluster3id = np.where(kmeans.labels_ == 3)[0].tolist()
    cluster4id = np.where(kmeans.labels_ == 4)[0].tolist()

    # simpan data review sesuai cluster
    cluster0text = "".join([clean_text[i] for i in cluster0id])
    cluster1text = "".join([clean_text[i] for i in cluster1id])
    cluster2text = "".join([clean_text[i] for i in cluster2id])
    cluster3text = "".join([clean_text[i] for i in cluster3id])
    cluster4text = "".join([clean_text[i] for i in cluster4id])

    # ngeplot wordcloud
    plt.figure(figsize=(8, 10))  # ngetest ukuran gambar
    wf = WordCloud(background_color='white', max_words=1000, random_state=153).generate(cluster0text)
    plt.imshow(wf)
    plt.savefig('static/image.png')
    plt.show()

    plt.figure(figsize=(8,10)) #ngetest ukuran gambar
    wf = WordCloud(background_color='white', max_words=1000,random_state=153).generate(cluster1text)
    plt.imshow(wf)
    plt.savefig('static/image2.png')
    plt.show()

    plt.figure(figsize=(8, 10))  # ngetest ukuran gambar
    wf = WordCloud(background_color='white', max_words=1000, random_state=153).generate(cluster2text)
    plt.imshow(wf)
    plt.savefig('static/image3.png')
    plt.show()

    plt.figure(figsize=(8, 10))  # ngetest ukuran gambar
    wf = WordCloud(background_color='white', max_words=1000, random_state=153).generate(cluster3text)
    plt.imshow(wf)
    plt.savefig('static/image4.png')
    plt.show()

    plt.figure(figsize=(8, 10))  # ngetest ukuran gambar
    wf = WordCloud(background_color='white', max_words=1000, random_state=153).generate(cluster4text)
    plt.imshow(wf)
    plt.savefig('static/image5.png')
    plt.show()


@app.route("/")
def home():
    jalan()
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)