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
    return metrics.accuracy_score(test_label, y_pred)


city_train_data, city_train_labels, city_train_sizes = parse_json('en_city_train.json')
city_test_data, city_test_labels, city_test_sizes = parse_json('en_city_test.json')
disease_train_data, disease_train_labels, disease_train_sizes = parse_json('en_disease_train.json')
disease_test_data, disease_test_labels, disease_test_sizes = parse_json('en_disease_test.json')

print(city_test_labels)
print("disease", disease_test_labels)

# doc_size - frequency graph

# plt.hist(city_train_sizes, bins=80, range=(0, 400))
# plt.show()
# plt.hist(disease_train_sizes, bins=80, range=(0, 400))
# plt.show()

"""
city 50 bin 
0-8 281
8-16 2322
16-24 1717
24-32 1618
32-40 1338
40-48 1058
48-56 908
56-64 834
64-72 703
72-80 628
80-88 616
88-96 696
96-104 729
104-112 553
112-120 364
120-128 462
128-136 553
136-144 668
144-152 557
152-160 280
160-168 239
168-176 199
176-184 203
184-192 180
192-200 168
200-208 116
208-216 131
216-224 109
224-232 86
232-240 85
240-248 87
248-256 76
256-264 67
264-272 64
272-280 60
280-288 54
288-296 55
296-304 36
304-312 40
312-320 43
320-328 33
328-336 32
336-344 29
344-352 28
352-360 29
360-368 27
368-376 21
376-384 19
384-392 17
392-400 22

"""
"""
city 80 bins
0-10 733
10-15 1578
15-20 1187
20-25 1038
25-30 1052
30-35 918
35-40 768
40-45 716
45-50 597
50-55 540
55-60 578
60-65 469
65-70 445
70-75 376
75-80 412
80-85 393
85-90 397
90-95 450
95-100 438
100-105 487
105-110 344
110-115 240
115-120 216
120-125 258
125-130 321
130-135 358
135-140 440
140-145 368
145-150 428
150-155 172
155-160 176
160-165 150
165-170 133
170-175 127
175-180 138
180-185 109
185-190 112
190-195 127
195-200 93
200-205 86
205-210 64
210-215 80
215-220 79
220-225 59
225-230 56
230-235 51
235-240 52
240-245 53
245-250 48
250-255 46
255-260 42
260-265 53
265-270 32
270-275 46
275-280 34
280-285 36
285-290 31
295-300 40
300-305 20
305-310 30
310-315 20
315-320 29
320-325 19
325-330 22
330-335 20
335-340 19
340-345 20
345-350 16
350-355 17
355-360 18
360-365 15
365-370 17
370-375 13
375-380 12
380-385 13
385-390 11
390-395 14
395-400 11
"""

city_sections = [
    [0, 0, 14],
    [1, 15, 24],
    [2, 25, 36],
    [3, 37, 54],
    [4, 55, 79],
    [5, 80, 104],
    [6, 105, 139],
    [7, 140, 194],
    [8, 195, 1000]
]
"""
disease 80 bin
0-10 440
10-15 740
15-20 907
20-25 898
25-30 960
30-35 949
35-40 956
40-45 873
45-50 832
50-55 803
55-60 728
60-65 708
65-70 665
70-75 570
75-80 579
80-85 508
85-90 488
90-95 456
95-100 415
100-105 413
105-110 383
110-115 346
115-120 295
120-125 340
125-130 281
130-135 264
135-140 226
140-145 205
145-150 214
150-155 187
155-160 206
160-165 195
165-170 157
170-175 169
175-180 119
180-185 135
185-190 127
190-195 118
195-200 123
200-205 111
205-210 94
210-215 86
215-220 79
220-225 69
225-230 73
230-235 51
235-240 59
240-245 50
245-250 56
250-255 52
255-260 55
260-265 49
265-270 38
270-275 40
275-280 39
280-285 28
285-290 25
295-300 19
300-305 27
305-310 33
310-315 18
315-320 30
320-325 24
325-330 23
330-335 9
335-340 19
340-345 17
345-350 10
350-355 17
355-360 14
360-365 9
365-370 10
370-375 13
375-380 12
380-385 9
385-390 7
390-395 16
395-400 11
"""
"""
disease
0-9 202
9-16 1141
16-24 1447
24-32 1528
32-40 1527
40-48 1368
48-56 1256
56-64 1153
64-72 1030
72-80 906
80-88 810
88-96 707
96-104 680
104-112 593
112-120 506
120-128 513
128-136 408
136-144 355
144-152 317
152-160 328
160-168 302
168-176 244
176-184 202
184-192 206
192-200 191
200-208 174
208-216 133
216-224 119
224-232 103
232-240 93
240-248 83
248-256 83
256-264 87
264-272 68
272-280 60
280-288 45
288-296 33
296-304 46
304-312 35
312-320 43
320-328 41
328-336 18
336-344 19
344-352 20
352-360 25
360-368 17
368-376 16
376-384 18
384-392 14
392-400 22
"""
disease_sections = [
    [0, 0, 23],
    [1, 24, 34],
    [2, 35, 48],
    [3, 49, 64],
    [4, 65, 84],
    [5, 85, 112],
    [6, 113, 152],
    [7, 153, 224],
    [8, 225, 1000]
]
disease_groups = [[], [], [], [], [], [], [], [], []]
disease_groups_label = [[], [], [], [], [], [], [], [], []]
disease = 0
while disease < len(disease_train_data):
    for size in disease_sections:
        if len(disease_train_data[disease]) <= size[2] and len(disease_train_data[disease]) >= size[1] :
            disease_groups[size[0]].append(disease_train_data[disease])
            disease_groups_label[size[0]].append(disease_train_labels[disease])
    disease = disease+1

city_groups = [[], [], [], [], [], [], [], [], []]
city_groups_label = [[], [], [], [], [], [], [], [], []]
city = 0
while city < len(city_train_data):
    for size in city_sections:
        if len(city_train_data[city]) <= size[2] and len(city_train_data[city]) >= size[1]:
            city_groups[size[0]].append(city_train_data[city])
            city_groups_label[size[0]].append(city_train_labels[city])
    city = city+1

print("Splitting is finished!!")

disease_accuracy_scores = []
disease_train = 0
while disease_train < len(disease_groups):
    disease_accuracy_scores.append(lda_and_SVC(disease_groups[disease_train], disease_groups_label[disease_train], disease_test_data, disease_test_labels, 27))
    disease_train = disease_train+1


city_accuracy_scores = []
city_train = 0
while city_train < len(city_groups):
    city_accuracy_scores.append(lda_and_SVC(city_groups[city_train], city_groups_label[city_train], city_test_data, city_test_labels, 30))
    city_train = city_train+1

for group_size in city_sections:
    print("city accuracy score between length ", group_size[1], "-", group_size[2], ":", city_accuracy_scores[group_size[0]])
for group_size in disease_sections:
    print("disease accuracy score between length ", group_size[1], "-", group_size[2], ":", disease_accuracy_scores[group_size[0]])

"""
city_dct, city_bow = create_bow(i for i in city_groups)
disease_dct, disease_bow = create_bow(i for i in disease_groups)

print("Preprocessing is finished!")
city_lda_results = []
for dct, bow in city_dct, city_bow:
    lda_model = LdaModel(corpus=bow, num_topics=30, id2word=dct)

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
