import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

''' wczytywanie danych '''

df = pd.read_csv('out/emotions-analysis.csv')
emo_an = pd.DataFrame(df)

X = emo_an['text'].values
y = emo_an['EMO_lab'].values

''' przygotowanie danych '''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

''' wektoryzacja - przekształcenie tekstów surowych na macierz cech TF-IDF '''

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1,2))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

''' klasyfikacja przy pomocy 4 modeli:
    SVC
    Regresja logistyczna
    Klasyfikator drzewa decyzyjnego
    Klasyfikator lasu losowego '''

clf_svm = SVC(kernel='linear', random_state=42)
clf_svm.fit(X_train_vectors, y_train)
y_pred_svm = clf_svm.predict(X_test_vectors)

clf_log = LogisticRegression(random_state=42)
clf_log.fit(X_train_vectors, y_train)
y_pred_log = clf_log.predict(X_test_vectors)

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train_vectors, y_train)
y_pred_tree = clf_tree.predict(X_test_vectors)

clf_forest = RandomForestClassifier(random_state=42)
clf_forest.fit(X_train_vectors, y_train)
y_pred_forest = clf_forest.predict(X_test_vectors)

''' ocena modeli '''

print('Trafność:')
print(clf_svm.score(X_test_vectors, y_test))
print(clf_log.score(X_test_vectors, y_test))
print(clf_tree.score(X_test_vectors, y_test))
print(clf_forest.score(X_test_vectors, y_test))

print('Precyzja:')
print(precision_score(y_test, y_pred_svm, average='macro', zero_division=0))
print(precision_score(y_test, y_pred_log, average='macro', zero_division=0))
print(precision_score(y_test, y_pred_tree, average='macro', zero_division=0))
print(precision_score(y_test, y_pred_forest, average='macro', zero_division=0))

print('Wrażliwość:')
print(recall_score(y_test, y_pred_svm, average='macro', zero_division=0))
print(recall_score(y_test, y_pred_log, average='macro', zero_division=0))
print(recall_score(y_test, y_pred_tree, average='macro', zero_division=0))
print(recall_score(y_test, y_pred_forest, average='macro', zero_division=0))

print('F1-score:')
print(f1_score(y_test, y_pred_svm, average='macro'))
print(f1_score(y_test, y_pred_log, average='macro'))
print(f1_score(y_test, y_pred_tree, average='macro'))
print(f1_score(y_test, y_pred_forest, average='macro'))

