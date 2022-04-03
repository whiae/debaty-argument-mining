import csv
import string
import pandas as pd
import spacy
from stop_words import get_stop_words

nlp = spacy.load('pl_spacy_model')

stop_words = get_stop_words('polish')

f = open("corpuses/polish.stopwords.txt", 'r', encoding='utf8')
polish_stopwords = []
lines = f.readlines()
for l in lines:
    polish_stopwords.append(l.strip())
f.close()

happy = {}
sadness = {}
anger = {}
fearness = {}
disgust = {}

data = {}

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
            sadness[word_2] = sad_2
            anger[word_2] = ang_2
            fearness[word_2] = fear_2
            disgust[word_2] = dis_2

labels = []
happy_values = []
anger_values = []
sadness_values = []
fear_values = []
disgust_values = []

df = pd.read_csv('past/premises.csv')
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

    tokens = []
    for token in nlp(i):
        if not token.lemma_ in polish_stopwords and not token.lemma_ in stop_words:
            tokens.append(token.lemma_)
            words_num = words_num + 1

    for token in tokens:
        if token in happy.keys():
            hap_val = hap_val + float(happy.get(str(token)))
            sad_val = sad_val + float(sadness.get(str(token)))
            angry_val = angry_val + float(anger.get(str(token)))
            fear_val = fear_val + float(fearness.get(str(token)))
            dis_val = dis_val + float(disgust.get(str(token)))

    happy_values.append(hap_val)
    anger_values.append(angry_val)
    sadness_values.append(sad_val)
    fear_values.append(fear_val)
    disgust_values.append(dis_val)

    maks = max(hap_val/words_num, sad_val/words_num, angry_val/words_num, fear_val/words_num, dis_val/words_num)
    if maks >= 0.5:
        if maks == hap_val/words_num:
            labels.append('HAP')
        elif maks == angry_val / words_num:
            labels.append('ANG')
        elif maks == sad_val/words_num:
            labels.append('SAD')
        elif maks == fear_val / words_num:
            labels.append('FEA')
        elif maks == dis_val / words_num:
            labels.append('DIS')
    else:
        labels.append('None')

emo_an['hap_val'] = happy_values
emo_an['ang_val'] = anger_values
emo_an['sad_val'] = sadness_values
emo_an['fea_val'] = fear_values
emo_an['dis_val'] = disgust_values
emo_an['EMO_lab'] = labels

emo_an.to_csv('emotions-analysis.csv', index=False, encoding='utf-8')