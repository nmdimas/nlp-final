import csv
import re
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np
import pandas
from IPython import get_ipython
import pymorphy3

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import spacy


def save_as_csv(clusters, filename):
    fields = ["#", "centroid", "top10"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in clusters:
            writer.writerow(row)


def get_stopwords_list():
    stop_file_path = 'stopword_ua.txt'
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))


def get_uk_name_list():
    path = 'uk_name.txt'
    with open(path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))


def get_custom_normal_form():
    path = 'custom_normal_form.txt'
    dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for words in lines:
            firsForm = ''
            for word in words.split():
                if firsForm == '':
                    firsForm = word
                    continue
                dict[word] = firsForm

        return dict


def to_clear(string):

    # str_oneline = re.sub('(\r\n|\n|\r)', ' ', string)

    # matchedProblem = re.match('.*Проблема:(.+)Передумова:.*', str_oneline)
    # matchedGoal = re.match(
    #     '.*Ціль, очікуваний результат:(.+)Задача на 1 урок:.*', str_oneline)

    # mathed_str = ''
    # if matchedProblem:
    #     mathed_str = matchedProblem.group(1)
    #     # print(matchedProblem.group(1))

    # if matchedGoal:
    #     mathed_str += ' ' + matchedGoal.group(1)
    #     # print(matchedGoal.group(1))
    # if mathed_str == '':
    #     return ''
    # string = mathed_str

    cleared_str = re.sub('[\W_]+', ' ', string.lower())
    cleared_str = re.sub('(^|\s)1\s', ' один ', cleared_str)
    cleared_str = re.sub('(^|\s)2\s', ' два ', cleared_str)
    cleared_str = re.sub('(^|\s)3\s', ' три ', cleared_str)
    cleared_str = re.sub('(^|\s)4\s', ' чотири ', cleared_str)
    cleared_str = re.sub('(^|\s)5\s', ' пять ', cleared_str)
    cleared_str = re.sub('(^|\s)6\s', ' шість ', cleared_str)
    cleared_str = re.sub('(^|\s)7\s', ' сім ', cleared_str)
    cleared_str = re.sub('(^|\s)8\s', ' вісім ', cleared_str)
    cleared_str = re.sub('(^|\s)9\s', ' девять ', cleared_str)
    cleared_str = re.sub('(^|\s)10\s', ' десять ', cleared_str)
    cleared_str = re.sub('(^|\s)11\s', ' одинадцять ', cleared_str)
    cleared_str = re.sub('(^|\s)12\s', ' дванядцять ', cleared_str)
    cleared_str = re.sub('(^|\s)вища математика\s',
                         ' вищаматематика ', cleared_str)

    cleared_str = re.sub('[\d_]+', ' ', cleared_str)  # remove special symbol
    cleared_str = re.sub(' +', ' ', cleared_str)  # Extra spaces

    return cleared_str.lower()


def save_cleared_dataset_to_csv(corpus, filename):
    fields = ["cleared"]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in corpus:
            writer.writerow({"cleared": row})


def mbkmeans_clusters(
    X,
    k,
    mb,
    print_silhouette_values,
):
    k_means_cluster = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(
            X, k_means_cluster.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[k_means_cluster.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return k_means_cluster, k_means_cluster.labels_


def vectorize(list_of_docs, model):

    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


morph = pymorphy3.MorphAnalyzer(lang='uk')
stop_words = get_stopwords_list()
ukranaian_names = get_uk_name_list()
custom_norm_form = get_custom_normal_form()

data_frame = pandas.read_csv('dataset_matema.csv', usecols=['Comment'])

nlp = spacy.load("uk_core_news_sm")
i = 0
corpus = []
docs = data_frame["Comment"].values
selectedDocs = []
for index, row in data_frame.iterrows():
    new_sent = []
    not_part = ''
    sentences = []

    doc = nlp(to_clear(row.Comment))
    for sent in doc.sents:
        word_list = sent.text.split()
        if len(word_list) < 5:  # Видаляємо короткі реченя, які меньше ніж розмір Bag Of Words
            continue
        selectedDocs.append(row.Comment)
        for word in word_list:
            if word in stop_words:
                continue
            if word in ukranaian_names:  # Уніфіковуємо всі імена в учень.
                word = 'учень'
            if word == 'не':   # Намагаємось обʼєднати не з дієсловами. Не встигає і встигає два принципово різні значення для домену
                not_part = 'не'
                continue
            if len(word) < 3:  # Маленькі слова (обрубки) відкидаємо
                continue
            # Зводимо різні написання в одне.
            word = morph.parse(word)[0].normal_form
            if word in custom_norm_form:
                word = custom_norm_form[word]

            # if p.tag.POS in ['NPRO', 'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']:
            #     not_part = ''
            #     continue
            # if len(word) == 3:
            #     print(word)
            if not_part:
                new_sent.append(not_part+word)
                not_part = ''
            else:
                new_sent.append(word)
        corpus.append(new_sent)
# Зберігаємо проміжні результати для аналізу
save_cleared_dataset_to_csv(corpus, 'cleared_corpus.csv')

model = Word2Vec(sentences=corpus, vector_size=1250,
                 window=5, min_count=3, workers=4)  # створюємо модель з розміром вектора 1250 (в фінальній версії в мене 1210 слів)
# Зберігаємо не бінарну версію моделі для зручного рендерінга
model.wv.save_word2vec_format('w2v_not_binnary', binary=False)
# Зберігаємо бінарну версію моделі для подальшого завантаження
model.save('w2v_binnary')

# model = Word2Vec.load("w2v_binnary")    # Можна швидко завантажувати вже готову модель

vectorized_docs = vectorize(corpus, model=model)   #
len(vectorized_docs), len(vectorized_docs[0])

num_cluster = 50
clustering, cluster_labels = mbkmeans_clusters(
    X=vectorized_docs,
    k=num_cluster,
    mb=1024,
    print_silhouette_values=True,
)

df_clusters = pandas.DataFrame({
    "text": selectedDocs,
    "tokens": [" ".join(text) for text in corpus],
    "cluster": cluster_labels
})

clusters_info = []
print("Most representative terms per cluster (based on centroids):")
for i in range(num_cluster):
    tokens_per_cluster = ""
    most_representative = model.wv.most_similar(
        positive=[clustering.cluster_centers_[i]], topn=10)
    centroid = clustering.cluster_centers_[i]
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    clusters_info.append({
        '#': i,
        'centroid': t[0],
        'top10': tokens_per_cluster
    })
    print(f"Cluster {i}: {tokens_per_cluster}")

save_as_csv(clusters_info, 'clustering.csv')
