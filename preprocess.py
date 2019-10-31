import json
import nltk
from nltk.tokenize import word_tokenize
from gensim.test.utils import datapath
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from gensim.corpora import Dictionary
from sklearn.utils import shuffle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

label2id = {}
label_count = 0


def parse(file):
    documents = []
    labels = []
    doc_lengths = []
    label_freq = {}
    with open(file, encoding="utf8") as json_file:
        data = json.load(json_file)
        for i in data:
            text = i['text']
            for section in i['annotations']:
                begin = section['begin']
                length = section['length']
                end = begin + length
                label = section['sectionLabel']

                if label not in label2id:
                    global label_count
                    label2id[label] = label_count
                    label_count += 1

                label_id = label2id[label]

                if label_id not in label_freq:
                    label_freq[label_id] = 0

                label_freq[label_id] += 1

                processed_text = preprocess(text[begin:end])
                doc_lengths.append(len(processed_text))
                documents.append(processed_text)
                labels.append(label_id)

    return documents, labels, doc_lengths, label_freq


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


train_data, train_labels, train_doc_lengths, train_label_freq = parse("wikisection_dataset_json/wikisection_en_disease_train.json")
test_data, test_labels, test_doc_lengths, test_label_freq = parse("wikisection_dataset_json/wikisection_en_disease_test.json")

en_disease_train = []
for i in range(19532):
    data = {}
    data['label_id'] = train_labels[i]
    data['size'] = train_doc_lengths[i]
    data['text'] = train_data[i]
    en_disease_train.append(data)

with open('en_disease_train.json', 'w') as fout:
    json.dump(en_disease_train, fout)

en_disease_test = []
for i in range(5501):
    data = {}
    data['label_id'] = test_labels[i]
    data['size'] = test_doc_lengths[i]
    data['text'] = test_data[i]
    en_disease_test.append(data)

with open('en_disease_test.json', 'w') as fout:
    json.dump(en_disease_test, fout)


train_data, train_labels, train_doc_lengths, train_label_freq = parse("wikisection_dataset_json/wikisection_en_city_train.json")
test_data, test_labels, test_doc_lengths, test_label_freq = parse("wikisection_dataset_json/wikisection_en_city_test.json")

print(train_labels)
# downsample

for key, val in train_label_freq.items():
    train_label_freq[key] = val / 92839 * 19532

for key, val in test_label_freq.items():
    test_label_freq[key] = val / 27301 * 5501

train_data, train_labels, train_doc_lengths = shuffle(train_data, train_labels, train_doc_lengths, random_state=0)
test_data, test_labels, test_doc_lengths = shuffle(test_data, test_labels, test_doc_lengths, random_state=0)

en_city_train = []
for i in range(92839):
    if (train_label_freq[train_labels[i]] != 0):
        data = {}
        data['label_id'] = train_labels[i]
        data['size'] = train_doc_lengths[i]
        data['text'] = train_data[i]
        en_city_train.append(data)
        train_label_freq[train_labels[i]] -= 1

with open('en_city_train.json', 'w') as fout:
    json.dump(en_city_train, fout)

en_city_test = []
for i in range(5501):
    if (test_label_freq[test_labels[i]] != 0):
        data = {}
        data['label_id'] = test_labels[i]
        data['size'] = test_doc_lengths[i]
        data['text'] = test_data[i]
        en_city_test.append(data)
        test_label_freq[test_labels[i]] -= 1

with open('en_city_test.json', 'w') as fout:
    json.dump(en_city_test, fout)
