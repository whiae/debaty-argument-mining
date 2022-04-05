import csv
import string
import pandas as pd
import spacy
from stop_words import get_stop_words
import matplotlib.pyplot as plt
import numpy as np

''' http://zil.ipipan.waw.pl/SpacyPL '''

nlp = spacy.load('pl_spacy_model')

''' wczytywanie stop words '''

stop_words = get_stop_words('polish')

f = open("corpuses/polish.stopwords.txt", 'r', encoding='utf8')
polish_stopwords = []
lines = f.readlines()
for l in lines:
    polish_stopwords.append(l.strip())
f.close()


''' wczytywanie korpusów '''

sentiment = {}

with open('corpuses/slownikWydzwieku.csv', 'r', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        sentiment[row[0]] = int(row[3])

with open('corpuses/negative_words_pl.txt', 'r', encoding='utf8') as csv_file:
    csv_reader1 = csv.reader(csv_file, delimiter=';')
    for row in csv_reader1:
        if row[0] not in sentiment.keys():
            sentiment[row[0]] = -1

with open('corpuses/declension_hatred_cleaned.csv', 'r', encoding='windows-1250') as csv_file:
    csv_reader2 = csv.reader(csv_file, delimiter=';')
    next(csv_reader2)
    for row in csv_reader2:
        if row[4] not in sentiment.keys():
            sentiment[row[4]] = -1

with open('corpuses/positive_words_pl.txt', 'r', encoding='utf8') as csv_file:
    csv_reader3 = csv.reader(csv_file, delimiter=';')
    for row in csv_reader3:
        if row[0] not in sentiment.keys():
            sentiment[row[0]] = 1


''' wczytywanie wyodrębnionych par konkluzja-przesłanka w formie tekstowej '''

df = pd.read_csv('original-model/pairs_text.csv')
sen_an = pd.DataFrame(df)


sentiment_values = []
labels = []
pos_count = 0
neg_count = 0
un_count = 0


for i in sen_an['text']:

    words_num = 0
    sentiment_val = 0

    for char in string.punctuation:
        i = i.replace(char, '').strip().lower()

    ''' pętla bierze tokeny z danej przesłanki i jeżeli nie należy on do stop words, dodaje do zbioru '''

    tokens = []
    for token in nlp(i):
        if not token.lemma_ in polish_stopwords and not token.lemma_ in stop_words:
            tokens.append(token.lemma_)
            words_num = words_num + 1

    ''' obliczanie wartości sentymentu dla tokenów w zbiorze '''

    for token in tokens:
        if token in sentiment.keys():
            sentiment_val = sentiment_val + sentiment.get(str(token))

    sentiment_values.append(sentiment_val)

    if sentiment_val != 0:
        if sentiment_val > 0:
            labels.append('POS')
            pos_count = pos_count + 1
        elif sentiment_val < 0:
            labels.append('NEG')
            neg_count = neg_count + 1
    else:
        labels.append('None')
        un_count = un_count + 1

''' zapisywanie danych '''

sen_an['SEN_val'] = sentiment_values
sen_an['SEN_lab'] = labels

''' zapisywanie do pliku '''

sen_an.to_csv('out/sentiment-analysis.csv', index=False, encoding='utf-8')


''' wizualizacja '''

np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots()

y_axis = ('POS', 'NEG', 'None')
x_axis = (pos_count, neg_count, un_count)

for i, v in enumerate(x_axis):
    ax.text(v + 1, i + 0.06, str(v), color='black', size='small')

plt.barh(y_axis, x_axis, align='center', color='green')
ax.invert_yaxis() # labels read top-to-bottom
ax.set_xlabel('Liczba przesłanek zakwalifikowanych do danej kategorii')
plt.savefig('plots/sen-an.png')