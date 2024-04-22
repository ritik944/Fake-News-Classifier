#fake news 
import pandas as pd
df=pd.read_csv('fake-news/train.csv')

df=df.dropna()

x=df.drop('label',axis=1)
y=df['label']

x.shape

y.shape

from keras.layers import Embedding
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras_preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense

voc_size=5000

#one hot rep

messages=x.copy()
messages.reset_index(inplace=True)
import nltk 
import re
from nltk.corpus import stopwords

#data preprossening


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

def corp(message):
    for i in range(0,len(message)):
        print(i)
        review=re.sub('[^a-zA-Z]',' ', message['text'][i])
        review=review.lower()
        review=review.split()
        review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
        review=' '.join(review)
        corpus.append(review)
    
corp(messages)    
onehot_rep=[one_hot(words,voc_size)for words in corpus]

##embedding represtation
sent_len=20
embedded_doc=pad_sequences(onehot_rep,padding='pre',maxlen=sent_len)
    
## ceating model

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

len(embedded_doc)

import numpy as np
x_final=np.array(embedded_doc)
y_final=np.array(y)

x_final.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_final,y_final,test_size=.33,random_state=25)

 
##model traing 

model.fit(x_final,y_final,validation_data=(x_test,y_test),epochs=10,batch_size=64)

##adding dropout

from keras.layers import Dropout
model=Sequential()
model.add(Embedding(voc_size, embedding_vector_features,input_length=sent_len))
model.add(Dropout(.3))
model.add(LSTM(100))
model.add(Dropout(.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


model.fit(x_final,y_final,validation_data=(x_test,y_test),epochs=10,batch_size=64)


# preformance metrics and accuracy 

y_pred=model.predict(x_test).round()
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
scorre=accuracy_score(y_test, y_pred)



    