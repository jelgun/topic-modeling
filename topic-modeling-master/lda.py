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

def lda_and_SVC(train_group, train_group_label, test_data, test_label, num_topics):

    dct, bow = create_bow(train_group)
    lda_model = LdaModel(
        corpus=bow,
        num_topics=num_topics,
        id2word=dct)
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
    clf.fit(document_topics, train_group_label)

    y_pred = clf.predict(t_document_topics)
    return metrics.accuracy_score(test_label, y_pred), metrics.f1_score(test_label, y_pred, average='micro'), y_pred


city_train_data, city_train_labels, city_train_sizes = parse_json('en_city_train.json')
city_test_data, city_test_labels, city_test_sizes = parse_json('en_city_test.json')
disease_train_data, disease_train_labels, disease_train_sizes = parse_json('en_disease_train.json')
disease_test_data, disease_test_labels, disease_test_sizes = parse_json('en_disease_test.json')


train_data = city_train_data + disease_train_data
train_labels = []
for i in range(len(train_data)):
    if i < len(city_train_data):
        train_labels.append(0)
    else:
        train_labels.append(1)

test_data = city_test_data + disease_test_data
test_labels = []

for i in city_test_data:
    test_labels.append(0)
for i in disease_test_data:
    test_labels.append(1)


train_sizes = city_train_sizes + disease_train_sizes
# doc_size - frequency graph


sizes_hist = plt.hist(train_sizes, bins=400, range=(0, 400))
plt.show()

data_groups = []
counter = 0
pre_station = 0
for size in range(len(sizes_hist[0])):
    size_num = sizes_hist[0][size]
    counter = counter + size_num
    if counter > 2000:
        counter = 0
        data_groups.append((len(data_groups), pre_station, size))
        pre_station = size+1

# doc_size - frequency graph

data_samples = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
data_samples_labels = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

for data in range(len(train_data)):
    for size in data_groups:
        if size[1] <= len(train_data[data]) <= size[2]:
            data_samples[size[0]].append(train_data[data])
            data_samples_labels[size[0]].append(train_labels[data])
print("Splitting is finished!!")
"""groups_value = []
groups_name = []
for i in data_samples:
    groups_value.append(len(i))
    groups_name.append(i[0])

city_numbers = []
disease_numbers = []
for i in data_samples_labels:
    city_number = 0
    disease_number = 0
    for label in i:
        if label == 0:
            city_number = city_number+1
        else:
            disease_number = disease_number+1
    city_numbers.append(city_number)
    disease_numbers.append(disease_number)
"""

data_accuracy_scores = []
data_f1_score = []
data_groups_pred = []

for data_train in range(len(data_samples)):
    accuracy, f1_score, y_pred = lda_and_SVC(data_samples[data_train], data_samples_labels[data_train], test_data, test_labels, 2)
    data_f1_score.append(f1_score)
    data_accuracy_scores.append(accuracy)
    data_groups_pred.append(y_pred)
plt.plot(data_accuracy_scores)
plt.show()

disease_true_numbers = []
city_true_numbers = []
for y_pred in range(len(data_groups_pred)):
    disease_true_number = 0
    city_true_number = 0
    for pred in range(len(data_groups_pred[y_pred])):
        if pred < 5501:
            if data_groups_pred[y_pred][pred] == 0:
                city_true_number = city_true_number+1
        else:
            if data_groups_pred[y_pred][pred] == 1:
                disease_true_number = disease_true_number + 1

    city_true_numbers.append(city_true_number)
    disease_true_numbers.append(disease_true_number)


disease_accuracy_scores = []
city_accuracy_scores = []
for i in range(len(data_groups)):
    city_accuracy_score = city_true_numbers[i]/5501
    disease_accuracy_score = disease_true_numbers[i]/5501
    disease_accuracy_scores.append(disease_accuracy_score)
    city_accuracy_scores.append(city_accuracy_score)

plt.plot(disease_accuracy_scores)
plt.show()
plt.plot(city_accuracy_scores)
plt.show()

"""
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
