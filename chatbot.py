<<<<<<< HEAD
import nltk
import json
import tflearn
import random
import pickle
from nltk.stem import WordNetLemmatizer
import numpy
import tensorflow
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('model.h5')

def clean_sentence(sentence):
	sentence_w = nltk.word_tokenize(sentence)
	sentence_w = [lemmatizer.lemmatize(word) for word in sentence_w]
	return sentence_w

def bag(sentence):
	sentence_w=clean_sentence(sentence)
	bag=[0]*len(words)
	for w in sentence_w:
		for i, word in enumerate(words):
			if word == w:
				bag[i]=1
	return numpy.array(bag)

def predict_class(sentence):
	bag_of_words = bag(sentence)
	res = model.predict(numpy.array([bag_of_words]))[0]
	Error_Threshold=0.25
	results=[[i,r] for i, r in enumerate(res) if r > Error_Threshold]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list=[]

	for r in results:
		return_list.append({'intents':classes[r[0]],'probability':str(r[1])})
	return return_list

def get_response(intents_l, intents_json):
	tag= intents_l[0]['intents']
	list_intents = intents_json['intents']
	for i in list_intents:
		if i['tag']==tag:
			result=random.choice(i['responses'])
			break
	return result
while True:
	message = input("User: ")
	ints = predict_class(message)
	res = get_response(ints, intents)
=======
import nltk
import json
import tflearn
import random
import pickle
from nltk.stem import WordNetLemmatizer
import numpy
import tensorflow
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('model.h5')

def clean_sentence(sentence):
	sentence_w = nltk.word_tokenize(sentence)
	sentence_w = [lemmatizer.lemmatize(word) for word in sentence_w]
	return sentence_w

def bag(sentence):
	sentence_w=clean_sentence(sentence)
	bag=[0]*len(words)
	for w in sentence_w:
		for i, word in enumerate(words):
			if word == w:
				bag[i]=1
	return numpy.array(bag)

def predict_class(sentence):
	bag_of_words = bag(sentence)
	res = model.predict(numpy.array([bag_of_words]))[0]
	Error_Threshold=0.25
	results=[[i,r] for i, r in enumerate(res) if r > Error_Threshold]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list=[]

	for r in results:
		return_list.append({'intents':classes[r[0]],'probability':str(r[1])})
	return return_list

def get_response(intents_l, intents_json):
	tag= intents_l[0]['intents']
	list_intents = intents_json['intents']
	for i in list_intents:
		if i['tag']==tag:
			result=random.choice(i['responses'])
			break
	return result
while True:
	message = input("User: ")
	ints = predict_class(message)
	res = get_response(ints, intents)
>>>>>>> 5a47c4bceb310f23d154b87e6259eff0e0a47ff4
	print("bot:"+res)