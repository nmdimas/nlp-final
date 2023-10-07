from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import re
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim
from matplotlib import pyplot as plt
import numpy as np
import pandas
from IPython import get_ipython
import pymorphy3

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer


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
    stop_file_path = 'uk_name.txt'
    with open(stop_file_path, 'r', encoding="utf-8") as f:
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


def to_clear_sent(string):
    new_sent = []
    not_part = ''
    for word in to_clear(string).split():
        if word in stop_words:
            continue
        if word in ukranaian_names:
            word = 'учень'
        if word == 'не':
            not_part = 'не'
            continue
        if len(word) < 3:
            continue
        p = morph.parse(word)[0]
        if not_part:
            new_sent.append(not_part+p.normal_form)
            not_part = ''
        else:
            new_sent.append(p.normal_form)
    return ' '.join(str(e) for e in new_sent)


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
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
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
    return km, km.labels_


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
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

def FunctionText2Vec(inpTextData):
    # Converting the text to numeric data
    X = vectorizer.transform(inpTextData)

    CountVecData = pandas.DataFrame(
        X.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

    # Creating empty dataframe to hold sentences
    W2Vec_Data = pandas.DataFrame()

    # Looping through each row for the data
    for i in range(CountVecData.shape[0]):

        # initiating a sentence with all zeros
        Sentence = np.zeros(1250)

        # Looping thru each word in the sentence and if its present in
        # the Word2Vec model then storing its vector
        for word in WordsVocab[CountVecData.iloc[i, :] >= 1]:
            # print(word)
            if word in model.key_to_index.keys():
                Sentence = Sentence + model[word]
        # Appending the sentence to the dataframe
        W2Vec_Data = pandas.concat(
            [W2Vec_Data, pandas.DataFrame([Sentence])], ignore_index=True)

    return (W2Vec_Data)

def FunctionPredictUrgency(model, inpText):
    X = FunctionText2Vec(inpText)
    X = PredictorScalerFit.transform(X)
    Prediction = model.predict(X)
    Result = pandas.DataFrame(data=inpText, columns=['Text'])
    Result['Prediction'] = Prediction
    return Prediction

morph = pymorphy3.MorphAnalyzer(lang='uk')

df = pandas.read_csv('dataset_matema.csv', usecols=['Comment', 'Tag'])

stop_words = get_stopwords_list()
ukranaian_names = get_uk_name_list()
custom_norm_form = get_custom_normal_form()

Comment = []
Tag = []
corpus = []
for ind in df.index:
    if not pandas.isnull(df.loc[ind, 'Tag']):
        new_sent = []
        not_part = ''
        for word in to_clear(df['Comment'][ind]).split():
            if word in stop_words:
                continue
            if word in ukranaian_names:
                word = 'учень'
            if word == 'не':
                not_part = 'не'
                continue
            if len(word) < 3:
                continue
            word = morph.parse(word)[0].normal_form
            if word in custom_norm_form:
                word = custom_norm_form[word]
            if not_part:
                new_sent.append(not_part+word)
                not_part = ''
            else:
                new_sent.append(word)
        corpus.append(new_sent)
        Tag.append(df['Tag'][ind])
        Comment.append(' '.join(str(e) for e in new_sent))


dfWithTags = pandas.DataFrame({'Comment': Comment, 'Tag': Tag})
# dfWithTags.groupby('Tag').size().plot(kind='bar')
# plt.show()

model = gensim.models.KeyedVectors.load_word2vec_format('w2v_not_binnary')

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(dfWithTags['Comment'].values)

CountVectorizedData = pandas.DataFrame(
    X.toarray(), columns=vectorizer.get_feature_names_out())
CountVectorizedData['Cluster'] = dfWithTags['Tag']

WordsVocab = CountVectorizedData.columns[:-1]

# Defining a function which takes text input and returns one vector for each sentence


W2Vec_Data = FunctionText2Vec(dfWithTags['Comment'])


W2Vec_Data.reset_index(inplace=True, drop=True)
W2Vec_Data['Tag'] = CountVectorizedData['Cluster']

DataForML = W2Vec_Data

TargetVariable = DataForML.columns[-1]
Predictors = DataForML.columns[:-1]

X = DataForML[Predictors].values
y = DataForML[TargetVariable].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=428)


PredictorScaler = MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit = PredictorScaler.fit(X)

# Generating the standardized values of X
X = PredictorScalerFit.transform(X)

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=428)

# GaussianNB is used in Binomial Classification
# MultinomialNB is used in multi-class classification
# clf = GaussianNB()
clf = MultinomialNB()
NB = clf.fit(X_train, y_train)
prediction = NB.predict(X_test)

# Measuring accuracy on Testing Data
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
print('Accuracy of the MultinomialNB on Testing Sample Data:', round(F1_Score, 2))

# Importing cross validation function from sklearn

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
Accuracy_Values = cross_val_score(NB, X, y, cv=5, scoring='f1_weighted')
print('\nAccuracy values for 5-fold Cross Validation:\n', Accuracy_Values)
print('\nFinal Average Accuracy of the model:',
      round(Accuracy_Values.mean(), 2))


clf = KNeighborsClassifier(n_neighbors=15)

# Creating the model on Training Data
KNN = clf.fit(X_train, y_train)
prediction = KNN.predict(X_test)

# Printing the Overall Accuracy of the model
F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(y_test, prediction))
print('Accuracy of the KNeighborsClassifier on Testing Sample Data:', round(F1_Score, 2))


clf = LogisticRegression(C=10, penalty='l2', solver='newton-cg')

# Printing all the parameters of logistic regression
# print(clf)

# Creating the model on Training Data
LOG = clf.fit(X_train, y_train)

# Generating predictions on testing data
prediction = LOG.predict(X_test)

# Printing sample values of prediction in Testing data
TestingData = pandas.DataFrame(data=X_test, columns=Predictors)
TestingData['Survived'] = y_test
TestingData['Predicted_Survived'] = prediction


# Printing the Overall Accuracy of the model
F1_Score = metrics.f1_score(y_test, prediction, average='weighted')
print(metrics.classification_report(y_test, prediction))
print(metrics.confusion_matrix(prediction, y_test))
print('Accuracy of the LogisticRegression on Testing Sample Data:', round(F1_Score, 2))

# Importing cross validation function from sklearn
# from sklearn.model_selection import cross_val_score

# Running 10-Fold Cross validation on a given algorithm
# Passing full data X and y because the K-fold will split the data and automatically choose train/test
# Accuracy_Values=cross_val_score(LOG, X , y, cv=10, scoring='f1_weighted')
# print('\nAccuracy values for 10-fold Cross Validation:\n',Accuracy_Values)
# print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(),2))


# Decision Trees
clf = tree.DecisionTreeClassifier(max_depth=20, criterion='gini')

DTree = clf.fit(X_train, y_train)
prediction = DTree.predict(X_test)


F1_Score = metrics.f1_score(y_test, prediction, average='weighted')

print('Accuracy of the DecisionTreeClassifier on Testing Sample Data:', round(F1_Score, 2))


# feature_importances = pandas.Series(
#    DTree.feature_importances_, index=Predictors)
# feature_importances.nlargest(10).plot(kind='barh')
#plt.show()


clf = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
FinalModel = clf.fit(X, y)

# Calling the function
NewTickets = [

    """Оцінка в школі - 9. Алгебра виходить краще, ніж геометрія. Математика потрібна для вступу в університет""",
    """Проблема: Мілана закінчила 6 клас, дівчата відмінниці, у них був рік репетитор, яка просто вирішувала за них дз, але не пояснювала, дівчатка розумні, але не було хорошого пояснення.

Передумова: репетитор не пояснювала і не вчила, нема розуміння.

Ціль, очікуваний результат: мати розуміння предмету.

Задача на 1 урок: діагностика 9 клас, контакт.

Характер дитини: розумна, відмінниця, лиш математика складно.""",
    "Нікіта закінчив 8 клас. Хоче повторити програму з 6 по 8 клас, виявити прогалини та підготуватись до 9 класу. потрібна підготовка до ДПА. Наголосив на проблемі з вирішенням квадратних порівнянь, проблеми виникають більше з аглеброю ніж з геометрією. приблизні оцінки були 8-9.",
    "Характер дитини: відмінно вчиться, але хочуть позайматися. Перша половина до 14 до кінця літа, а потім після обіду вже." ,
    "не цікаво йому зовсім, нема мотивації, любить комп ігри, якщо можна, то зацікавити тим що математика потрібна у іграх і тд, давати цікаві завдання, головне - зацікавити"
]


FinalModel = clf

for tiket in NewTickets:
    print(tiket)
    clear_tiket = to_clear_sent(tiket)

    print('Prediction : ' + FunctionPredictUrgency(FinalModel, inpText=[clear_tiket]))
    X = FunctionText2Vec([clear_tiket])
    X = PredictorScalerFit.transform(X)

    probability = FinalModel.predict_proba(X)
    dataFrame = pandas.DataFrame(
        {'Probability': probability[0], 'Tag': clf.classes_})
    np.set_printoptions(suppress=True)
    with pandas.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 10,
                               ):
        print(dataFrame.sort_values(by='Probability', ascending=False))
    print('- - - - - - - - - - - - - ')
