import json
import pandas as pd
import glob
import random

maps = glob.glob("maps/*.json")

'''
    conclusionPremiseDict Create dictionary of pairs with an identifier with the following form: 
    {id: {"conclusion": <SINGLE_CONCLUSION>, "premises":[<LIST_OF_PREMISES>]}}
'''


def conclusionPremiseDict(premises, conclusions):
    pairs = {}
    for i, x in enumerate(conclusions):
        pairs[i] = {'conclusion': x, 'premises': []}
        id_to = x['fromID']
        for p in premises:
            if p['toID'] == id_to:
                pairs[i]['premises'].append(p)

    return pairs


'''
    aduPairs create list of ADU pairs containing connected conclusion and premise [[conclusion, premise]] 
'''


def aduPairs(edgePairs, nodesById):
    aduPair = []
    for pair in edgePairs.values():
        for p in pair['premises']:
            aduPair.append([nodesById[pair['conclusion']['toID']]['text'], nodesById[p['fromID']]['text']])
    return (aduPair)


'''
    pairs creates conclusion - premise pairs for one map
'''


def pairs(map):
    with open(map) as f:
        data = json.loads(f.read())
    # Creating nodesById dictionary which has nodeID as key and whole node as value for more efficient data extraction.
    nodesById = {}
    for _, node in enumerate(data['nodes']):
        nodesById[node['nodeID']] = node
    # Premises are nodes that have ingoing edges that are type 'RA' and outgoing edges that are type 'I'.
    premises = [x for x in data['edges'] if
                nodesById[x['fromID']]['type'] == 'I' and nodesById[x['toID']]['type'] == 'RA']

    # Conclusions are nodes that have ingoing edges that are type 'I' and outgoing edges that are type 'RA'.
    conclusions = [x for x in data['edges'] if
                   nodesById[x['toID']]['type'] == 'I' and nodesById[x['fromID']]['type'] == 'RA']
    edgePairs = conclusionPremiseDict(premises, conclusions)
    adus = aduPairs(edgePairs, nodesById)
    return adus, conclusions, premises, nodesById


'''
    comb makes combination of conclusions and premises lists and returns list of pairs that are not conclusion-premise pairs 
'''


def comb(conclusions, premises, l, nodesById):
    combList = [(x, y) for x in conclusions for y in premises]
    smallCombList = []
    for _ in range(l):
        p = random.choice(combList)
        smallCombList.append([nodesById[p[0]['toID']]['text'], nodesById[p[1]['fromID']]['text']])
    return smallCombList


'''
    truePairs is list of all conclusion-premise pairs; falsePairs is list od conclusion-premise non pairs
'''
truePairs = []
conclusions = []
premises = []
nodesById = {}

for m in maps:
    adus, c, p, n = pairs(m)
    truePairs.extend(adus)
    conclusions.extend(c)
    premises.extend(p)
    nodesById = {**nodesById, **n}

falsePairs = comb(conclusions, premises, len(truePairs), nodesById)

df = pd.DataFrame(truePairs)
df1 = df.dropna()
df[1].to_csv('premises.csv', header = ['text'], index=False, encoding='utf-8')



#
# samples = df3['sim']
# labels = df3['EMO_lab']
#
# trainSamples = np.array(samples[:len(samples)//2])
# trainLabels = np.array(labels[:len(samples)//2])
# testSamples = np.array(samples[len(samples)//2:])
# testLabels = np.array(labels[len(labels)//2:])
#
# model = tf.keras.Sequential([
#     Dense(units=16, input_shape=(1,), activation='sigmoid'),
#     Dense(units=32, activation='relu'),
#     Dense(units=64, activation='relu'),
#     Dense(units=5, activation='softmax')
# ])
# model.summary()
#
# '''
#   train model
# '''
# model.compile(optimizer=Adam(learning_rate=0.0007), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x=trainSamples, y=trainLabels, validation_split=0.1, batch_size=15, epochs=100, shuffle=True, verbose=2)
#
# '''
#   predictions
# '''
# predictions = model.predict(x=testSamples, batch_size=10, verbose=0)
#
# roundedPredictions = np.argmax(predictions, axis=-1)
#
# cm = confusion_matrix(y_true = testLabels, y_pred = roundedPredictions)
#
#
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix,', cmap=mpl.cm.Greens):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:np.newaxis]
#         print('Normalized confusiom matrix')
#     else:
#         print('Confusion matrix without normalization')
#
#     print(cm)
#
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('Prawdziwe')
#     plt.xlabel('Przewidywane')
#     plt.show()
#
# cm_plot_labels = ['POS', 'NEG']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Macierz pomyłek")