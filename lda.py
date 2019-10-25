"""
[+]Preprocessing:
    [+]1) Parse files
    [+]2) Tokenization
    [+]3) Remove stopwords, numbers, punctuation
    [+]4) Lemmatization
    [+]5) Stemming
[+]Create BOW, remove the most and least frequent words, keep ~100k words
[+]Apply LDA
[+]Evaluate:
    [+]1) Topic Coherence Metric (CV)
    [+]2) tSNE Visualization
"""
import json
import nltk
from nltk.tokenize import word_tokenize
from gensim.test.utils import datapath
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.manifold import TSNE
from sklearn import svm, metrics
import numpy as np
from matplotlib import pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

NUM_TOPICS = 30

label2id = {}
label_id = 0


def parse(file):
    documents = []
    labels = []
    with open(file, encoding="utf8") as json_file:
        data = json.load(json_file)
        for i in data:
            text = i['text']
            for section in i['annotations']:
                begin = section['begin']
                end = begin + section['length']
                label = section['sectionLabel']

                if label not in label2id:
                    global label_id
                    label2id[label] = label_id
                    label_id += 1

                documents.append(preprocess(text[begin:end]))
                labels.append(label2id[label])

    return documents, labels


def preprocess(data):
    # tokenize
    tokenized_data = [i.lower() for i in word_tokenize(data)]

    # remove stopwords, numbers and punctuations
    filtered_data = [w for w in tokenized_data
                     if (w not in stop_words) and
                     (not w.isdigit()) and
                     (w not in punctuation)]

    # lemmatize
    lemm_data = [lemmatizer.lemmatize(w) for w in filtered_data]

    # stem
    stem_data = [ps.stem(w) for w in lemm_data]

    return stem_data


def create_bow(data):
    dct = Dictionary(data)
    dct.filter_extremes(no_below=20)
    bow = [dct.doc2bow(doc) for doc in data]
    return dct, bow


train_data, train_labels = parse("wikisection_dataset_json/wikisection_en_city_train.json")
test_data, test_labels = parse("wikisection_dataset_json/wikisection_en_city_test.json")
dct, bow = create_bow(train_data)
print("Preprocessing is finished!")

histogram = plt.hist(train_labels, normed=True)
plt.show(histogram)

"""lda_model = LdaModel(
    corpus=bow,
    num_topics=NUM_TOPICS,
    id2word=dct)
print("Lda is finished!")

document_topics = []
for doc_bow in bow:
    ls = []
    for top, prob in lda_model.get_document_topics(bow=doc_bow, minimum_probability=0.0):
        ls.append(prob)
    document_topics.append(ls)

t_document_topics = []
for doc in test_data:
    doc_bow = dct.doc2bow(doc)
    ls = []
    for top, prob in lda_model.get_document_topics(bow=doc_bow, minimum_probability=0.0):
        ls.append(prob)
    t_document_topics.append(ls)

clf = svm.LinearSVC()
clf.fit(document_topics, train_labels)

y_pred = clf.predict(t_document_topics)
print(metrics.accuracy_score(test_labels, y_pred))



# Topic Coherence
coherence_model_lda = CoherenceModel(
    model=lda_model,
    texts=train_data,
    corpus=bow,
    dictionary=dct,
    coherence='c_v',
    processes=1)

coherence = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence)


X = np.zeros((len(bow), NUM_TOPICS))
for i in range(len(bow)):
    for (k, y) in lda_model[bow[i]]:
        X[i][k] = y

topic_num = np.argmax(X, axis=1)

colors = np.random.rand(NUM_TOPICS)
doc_colors = []
for i in range(X.shape[0]):
    doc_colors.append(colors[topic_num[i]])

tsne_embedding = TSNE(random_state=0).fit_transform(X)

plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=doc_colors)
plt.show()
"""
