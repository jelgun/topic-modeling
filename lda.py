import json
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.manifold import TSNE
from sklearn import svm, metrics
import numpy as np
from matplotlib import pyplot as plt


def create_bow(data):
    dct = Dictionary(data)
    dct.filter_extremes(no_below=20)
    bow = [dct.doc2bow(doc) for doc in data]
    return dct, bow


def parse_json(file):
    with open(file, encoding="utf8") as json_file:
        json_data = json.load(json_file)
        text_data = []
        labels = []
        sizes = []
        for i in json_data:
            text_data.append(i['text'])
            labels.append(i['label_id'])
            sizes.append(int(i['size']))

        return text_data, labels, sizes
        
city_train_data, city_train_labels, city_train_sizes = parse_json('en_city_train.json')
city_test_data, city_test_labels, city_test_sizes = parse_json('en_city_test.json')
disease_train_data, disease_train_labels, disease_train_sizes = parse_json('en_disease_train.json')
disease_test_data, disease_test_labels, disease_test_sizes = parse_json('en_disease_test.json')

# doc_size - frequency graph
plt.hist(city_train_sizes, bins=200, range=(0, 300))
plt.show()
"""
doc_freq = {}
for i in city_train_sizes:
    if i not in doc_freq:
        doc_freq[i] = 0
    else:
        doc_freq[i] += 1

"""
"""
dct, bow = create_bow(train_data)
print("Preprocessing is finished!")

lda_model = LdaModel(                                                           
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
