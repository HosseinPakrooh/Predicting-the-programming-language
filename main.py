import csv
import nltk
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier






def rem_html_tags(body):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', body)

train = []
test = []
mytrainY = []
mytestY = []


df = pd.read_csv('41841538578713.csv', encoding="utf8")
a = 0

for row in range(0, len(df)):
    # print(df.loc[row,'post'])
    clean_row = rem_html_tags(df.loc[row, 'post'])

    if (df.loc[row, 'tags'] == 'ios'):
        train.append(clean_row)
        mytrainY.append(0)

    if (df.loc[row, 'tags'] == 'python'):
        train.append(clean_row)
        mytrainY.append(1)
    if (df.loc[row, 'tags'] == 'c++'):
        train.append(clean_row)
        mytrainY.append(2)
    if (df.loc[row, 'tags'] == 'java'):
        train.append(clean_row)
        mytrainY.append(3)
    if (df.loc[row, 'tags'] == 'php'):
        train.append(clean_row)
        mytrainY.append(4)




temp_train = train
temp_train_y = mytrainY


train = temp_train[:8000]
mytrainY = temp_train_y[:8000]
test = temp_train[8000:]
mytestY = temp_train_y[8000:]






SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
def preprocess(np):
    x = [SPACE.sub(" ", line) for line in np]
    return x
train = preprocess(train)
test = preprocess(test)

################################################################Lemmatizer
def Lemmatizer(cp):
    R = []
    for row in cp:
        R.append(WordNetLemmatizer().lemmatize(row))
    return R
train_data = Lemmatizer(train)
test_data = Lemmatizer(test)
print('Lemmatizer')
#################################################################Porterstemmer
def porterstemmer(cp):
    R = []
    for row in cp:
        R.append(PorterStemmer().stem(row))
    return R
train_data = porterstemmer(train)
test_data = porterstemmer(test)
print('Porterstemmer')
print(len(test_data[0]))
print(len(test_data[23]))
print(len(test_data[56]))
################################################################Stopwords
def stopW(cp):
    R = []
    StopWords = stopwords.words('english')
    sw = ['i', 'be', 'been', 'being','a', 'an', 'the', 'and', 'but', 'if',
          'or', 'as', 'while', 'of', 'at', 'by', 'for','with', 'between',
          'into','before', 'after', 'above', 'below', 'to', 'from', 'up',
          'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again','then',
          'once', 'all', 'any', 'other', 'no', 'not', 'only','than','s', 't',
          'd', 'll', 'm', 'o', 'this', 'is', 'do']
    MeStopWords = [word for word in StopWords if word not in sw]
    for row in cp:
        R.append(' '.join([word for word in row.split() if word not in MeStopWords]))
    return R

train_data = stopW(train_data)
test_data = stopW(test_data)
print('Stopwords')
print(len(test_data[0]))
print(len(test_data[23]))
print(len(test_data[56]))


vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 5), min_df=2)
vectorizer_fit = vectorizer.fit_transform(train_data)
dataframe = pd.DataFrame(vectorizer_fit.todense(), columns=vectorizer.get_feature_names())


vectorizer_test = vectorizer.transform(test_data)
dataframetest = pd.DataFrame(vectorizer_test.todense())

X_train = dataframe.to_numpy()
X_test = dataframetest.to_numpy()
Y_train = mytrainY
Y_test = mytestY


print('TF-idf')
print('MultinomialNB')
clf = MultinomialNB()
clf.fit(X_train, Y_train)
print("Accuracy Train:")
print(clf.score(X_train, Y_train))
print("Accuracy Test:")
print(clf.score(X_test, Y_test))
yp = clf.predict(X_test)
# print(clf.predict(vectorizer.transform([])))
precision = precision_score(Y_test, yp, average="macro")
print("precision: ", precision)
recall = recall_score(Y_test, yp, average="macro")
print("recall: ", recall)
f1 = f1_score(Y_test, yp, average="macro")
print("f1: ", f1)
print(classification_report(Y_test, yp, labels=[0, 1, 2, 3, 4]))


print('###########################')


print('RidgeClassifier')
scikit_log_reg = LogisticRegression(solver='liblinear')
model = scikit_log_reg.fit(X_train, Y_train)
print("Accuracy  on Train:")
print(model.score(X_train, Y_train))
print("Accuracy  on Test:")
print(model.score(X_test, Y_test))
yp = model.predict(X_test)
precision = precision_score(Y_test, yp, average ="macro")
print("precision: ", precision)
recall = recall_score(Y_test, yp, average ="macro")
print("recall: ", recall)
f1 = f1_score(Y_test, yp, average="macro")
print("f1: ", f1)
print(classification_report(Y_test, yp, labels=[0, 1, 2, 3, 4]))



print('###########################')



print('###########################')


print('MLPClassifier')
MLPClassifier()
classifier = MLPClassifier(hidden_layer_sizes=(100, 5))
classifier.fit(X_train, Y_train)
print("Accuracy Train:")
print(classifier.score(X_test, Y_test))
print("Accuracy Test:")
print(classifier.score(X_test, Y_test))
yp = classifier.predict(X_test)
precision = precision_score(Y_test, yp, average="macro")
print("precision: ", precision)
recall = recall_score(Y_test, yp, average="macro")
print("recall: ", recall)
f1 = f1_score(Y_test, yp, average="macro")
print("f1: ", f1)
print(classification_report(Y_test, yp, labels=[0, 1, 2, 3, 4]))


print('###########################')


print('SVM')
final_svm_ngram = LinearSVC(C=0.05)
final_svm_ngram.fit(X_train, Y_train)
print("Accuracy Train:")
print(final_svm_ngram.score(X_train, Y_train))
print("Accuracy Test:")
print(final_svm_ngram.score(X_test, Y_test))
yp = final_svm_ngram.predict(X_test)
precision = precision_score(Y_test, yp, average="macro")
print("precision: ", precision)
recall = recall_score(Y_test, yp, average="macro")
print("recall: ", recall)
f1 = f1_score(Y_test, yp, average="macro")
print("f1: ", f1)
print(classification_report(Y_test, yp, labels=[0, 1, 2, 3, 4]))

print('###########################')


neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, Y_train)
print("KNN")
print(neigh.score(X_train, Y_train))
print("Accuracy Test:")
print(neigh.score(X_test, Y_test))
yp = neigh.predict(X_test)
precision = precision_score(Y_test, yp, average="macro")
print("precision: ", precision)
recall = recall_score(Y_test, yp, average="macro")
print("recall: ", recall)
f1 = f1_score(Y_test, yp, average="macro")
print("f1: ", f1)
print(classification_report(Y_test, yp, labels=[0, 1, 2, 3, 4]))
















