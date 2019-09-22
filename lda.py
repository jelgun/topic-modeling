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
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def parse(files):
    documents = []
    for file in files:
        with open(file) as json_file:
            data = json.load(json_file)
            for i in data:
                documents.append(i['text'])
    return documents


def preprocess():
    files = ['wikisection_dataset_json/wikisection_en_city_train.json',
             'wikisection_dataset_json/wikisection_en_disease_train.json']
    documents = parse(files)

    # tokenize
    tokenized_data = []
    for data in documents:
        tokenized_data.append(i.lower() for i in word_tokenize(data))

    # remove stopwords, numbers and punctuations
    stop_words = set(stopwords.words('english'))
    filtered_data = []
    for data in tokenized_data:
        filtered_data.append([w for w in data
                             if (w not in stop_words) and
                             (not w.isdigit()) and
                             (w not in punctuation)])

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemm_data = []
    for data in filtered_data:
        lemm_data.append([lemmatizer.lemmatize(w) for w in data])

    # stem
    ps = PorterStemmer()
    stem_data = []
    for data in lemm_data:
        stem_data.append([ps.stem(w) for w in data])

    return stem_data


def create_bow(data):
    dct = Dictionary(data)
    dct.filter_extremes(no_below=20)
    bow = [dct.doc2bow(doc) for doc in data]
    return dct, bow


data = preprocess()
dct, bow = create_bow(data)
lda_model = LdaMulticore(
    corpus=bow,
    num_topics=55,
    id2word=dct,
    passes=5,
    workers=2)

# word weights for topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Topic Coherence
coherence_model_lda = CoherenceModel(
    model=lda_model,
    texts=data,
    corpus=bow,
    dictionary=dct,
    coherence='c_v')

coherence = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence)

# tsne visualization
X = np.zeros((len(bow), NUM_TOPICS))
for i in range(len(bow)):
    for (k, y) in lda_model[bow[i]]:
        X[i][k] = y

topic_num = np.argmax(X, axis=1)

colors = np.random.rand(NUM_TOPICS)
doc_colors = []
for i in range(X.shape[0]):
    doc_colors.append(colors[topic_num[i]])

tsne_embedding = TSNE(
    n_components=2,
    random_state=0,
    init='pca').fit_transform(X)

plt.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=doc_colors)
plt.show()
