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

# plt.hist(city_train_sizes, bins=50, range=(0, 400))
# plt.hist(disease_train_sizes, bins=50, range=(0,400))

city_sections = [
    (0, 0, 8),
    (1, 9, 15),
    (2, 16, 31),
    (3, 32, 40),
    (4, 41, 47),
    (5, 48, 63),
    (6, 64, 103),
    (7, 104, 111),
    (8, 112, 119),
    (9, 120, 135),
    (10, 136, 143),
    (11, 144, 151),
    (12, 152, 167),
    (13, 168, 199),
    (14, 200, 223),
    (15, 224, 247),
    (16, 248, 255),
    (17, 256, 279),
    (18, 280, 295),
    (19, 296, 303),
    (20, 304, 319),
    (21, 320, 367),
    (22, 368, 391),
    (23, 392, 399),
    (24, 400, 1000)
]

disease_sections = [
    (0, 0, 7),
    (1, 7, 15),
    (2, 16, 23),
    (3, 24, 39),
    (4, 40, 55),
    (5, 56, 71),
    (6, 72, 87),
    (7, 88, 103),
    (8, 104, 111),
    (9, 112, 127),
    (10, 128, 143),
    (11, 144, 167),
    (12, 168, 175),
    (13, 176, 199),
    (14, 200, 207),
    (15, 208, 223),
    (16, 224, 239),
    (17, 240, 263),
    (18, 264, 279),
    (19, 280, 327),
    (20, 328, 335),
    (21, 336, 343),
    (22, 344, 359),
    (23, 360, 391),
    (24, 392, 399),
    (25, 400, 1000)
]

for group_no, left_border, right_border in disease_sections:
    print(group_no, left_border, right_border)



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
