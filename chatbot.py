import nltk
import json
import tflearn
import random
import pickle
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

Lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('model.h5')
model._make_predict_function()


def bag(sentence):
	cleaned_sentence = nltk.word_tokenize(sentence)
	cleaned_sentence = [Lemmatizer.lemmatize(word) for word in cleaned_sentence]
	bag=[0]*len(words)
	for w in cleaned_sentence:
		for i, word in enumerate(words):
			if word == w:
				bag[i]=1
	return np.array(bag)

def predict_and_respond(sentence):
    p = bag(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    try:
        tag = return_list[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag']  == tag:
                result = random.choice(i['responses'])
                break
    except:
    	result = "Sorry, What do you mean?"
    return result
# while True:
# 	message = input("User: ")
# 	ints = predict_class(message)
# 	res = get_response(ints, intents)

# 	print("bot:"+res)
