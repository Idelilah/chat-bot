import nltk
from nltk.stem import WordNetLemmatizer
import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import json
import random
import pickle

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words=[]
classes=[]
documents=[]

for intent in intents['intents']:
	for pattern in intent['patterns']: 
		word_list=nltk.word_tokenize(pattern)
		words.extend(word_list)
		documents.append((word_list, intent['tag']))
		if intent['tag'] not in classes:
			classes.append(intent['tag'])
ignore=[".","?",":)", "!"]
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
out_empty = [0]*len(classes)

for document in documents:
	bag=[]
	word_patterns=document[0]
	word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
	for word in words:
		bag.append(1) if word in word_patterns else bag.append(0)

	output_row=list(out_empty)
	output_row[classes.index(document[1])] = 1
	training.append([bag, output_row])

random.shuffle(training)
training=numpy.array(training)

training_x = list(training[:,0])
training_y= list(training[:,1])


model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]),activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(numpy.array(training_x), numpy.array(training_y),epochs=200,batch_size=8, verbose=1)
model.save('model.h5',hist)