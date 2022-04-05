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

happy = {}
sadness = {}
anger = {}
fearness = {}
disgust = {}

with open('corpuses/nawl_emo_cat.csv', 'r', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        word = row[2]
        hap = row[4].replace(',', '.')
        angry = row[6].replace(',', '.')
        sad = row[8].replace(',', '.')
        fear = row[10].replace(',', '.')
        dis = row[12].replace(',', '.')
        happy[word] = hap
        anger[word] = angry
        sadness[word] = sad
        fearness[word] = fear
        disgust[word] = dis

with open('corpuses/emean_emo_cat.csv', 'r', encoding='utf8') as csv_file_2:
    csv_reader_2 = csv.reader(csv_file_2, delimiter=',')
    next(csv_reader_2)
    for row in csv_reader_2:
        word_2 = row[1]
        hap_2 = row[11].replace(',', '.')
        ang_2 = row[6].replace(',', '.')
        sad_2 = row[9].replace(',', '.')
        fear_2 = row[8].replace(',', '.')
        dis_2 = row[7].replace(',', '.')
        if word_2 not in happy.keys():
            happy[word_2] = hap_2
            anger[word_2] = ang_2
            sadness[word_2] = sad_2
            fearness[word_2] = fear_2
            disgust[word_2] = dis_2

labels = []
happy_values = []
anger_values = []
sadness_values = []
fear_values = []
disgust_values = []

hap_count = 0
ang_count = 0
sad_count = 0
fea_count = 0
dis_count = 0
none_count = 0

df = pd.read_csv('original-model/pairs_text.csv')
emo_an = pd.DataFrame(df)

for i in emo_an['text']:

    words_num = 0
    hap_val = 0
    sad_val = 0
    angry_val = 0
    fear_val = 0
    dis_val = 0

    for char in string.punctuation:
        i = i.replace(char, '').strip().lower()

    ''' pętla bierze tokeny z danej przesłanki i jeżeli nie należy on do stop words, dodaje do zbioru '''

    tokens = []
    for token in nlp(i):
        if not token.lemma_ in polish_stopwords and not token.lemma_ in stop_words:
            tokens.append(token.lemma_)
            words_num = words_num + 1

    ''' obliczanie wartości emocji dla słów w zbiorze '''


    for token in tokens:
        if token in happy.keys():
            hap_val = hap_val + float(happy.get(str(token)))
            angry_val = angry_val + float(anger.get(str(token)))
            sad_val = sad_val + float(sadness.get(str(token)))
            fear_val = fear_val + float(fearness.get(str(token)))
            dis_val = dis_val + float(disgust.get(str(token)))

    hap_ro = round((hap_val/words_num), 2)
    ang_ro = round((angry_val/words_num), 2)
    sad_ro = round((sad_val/words_num), 2)
    fea_ro = round((fear_val/words_num), 2)
    dis_ro = round((dis_val/words_num), 2)

    happy_values.append(hap_ro)
    anger_values.append(ang_ro)
    sadness_values.append(sad_ro)
    fear_values.append(fea_ro)
    disgust_values.append(dis_ro)

    ''' przypisywanie odpowiedniej kategorii zależnie od najwyższej wartości emocji '''

    maks = max(hap_ro, ang_ro, sad_ro, fea_ro, dis_ro)
    if maks > 0:
        if maks == hap_ro:
            labels.append('HAP')
            hap_count = hap_count + 1
        elif maks == ang_ro:
            labels.append('ANG')
            ang_count = ang_count + 1
        elif maks == sad_ro:
            labels.append('SAD')
            sad_count = sad_count + 1
        elif maks == fea_ro:
            labels.append('FEA')
            fea_count = fea_count + 1
        elif maks == dis_ro:
            labels.append('DIS')
            dis_count = dis_count + 1
    else:
        labels.append('None')
        none_count = none_count + 1

''' zapisywanie danych '''

emo_an['hap_val'] = happy_values
emo_an['ang_val'] = anger_values
emo_an['sad_val'] = sadness_values
emo_an['fea_val'] = fear_values
emo_an['dis_val'] = disgust_values
emo_an['EMO_lab'] = labels

''' zapisywanie do pliku '''

emo_an.to_csv('out/emotions-analysis.csv', index=False, encoding='utf-8')


''' wizualizacja '''

np.random.seed(19680801)
plt.rcdefaults()
fig, ax = plt.subplots()

y_axis = ('HAP', 'ANG', 'SAD', 'FEA', 'DIS', 'None')
x_axis = (hap_count, ang_count, sad_count, fea_count, dis_count, none_count)

for i, v in enumerate(x_axis):
    ax.text(v + 1, i + 0.06, str(v), color='black', size='small')

plt.barh(y_axis, x_axis, align='center', color='maroon')
ax.invert_yaxis() # labels read top-to-bottom
ax.set_xlabel('Liczba przesłanek zakwalifikowanych do danej kategorii')
plt.savefig('plots/emo-an.png')