#encoding=utf-8
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk  #用来分词
import collections  #用来统计词频
import numpy as np
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

maxlen = 0  #句子最大长度,最长的句子有多少个单词
word_freqs = collections.Counter()  #词频，列表中记录每个单词出现的次数
num_recs = 0 # 样本数

with open('./dns5.txt','r+',encoding='UTF-8') as f:
    for line in f:
        sentence ,label = line.strip().split(",")
        words = list(sentence)
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}


X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i=0
with open('./dns5.txt','r+',encoding='UTF-8') as f:
    for line in f:
        sentence, label = line.strip().split(",")
        words = list(sentence)
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs

        y[i] = int(label)
        i += 1

X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)


EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 3

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
#BATCH_SIZE = 32
BATCH_SIZE = 5000
#NUM_EPOCHS = 10
NUM_EPOCHS = 50
clf = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))


model.save('my_model5000.h5')


score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]
    sent = "".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))


model = load_model('my_model5000.h5')


INPUT_SENTENCES = ['nylalbobhyhirgh','oyqlwhmbyoseeyfg','sina', 'baidu','ilovethisgame','zxcvbnmkj']
XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in  INPUT_SENTENCES:

    words = list(sentence)
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX) ]
label2word = {0:'正常', 1:'恶意'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))