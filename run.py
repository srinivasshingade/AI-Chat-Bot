
# Installing Dependencies for  NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# Restore  our data structures
import pickle
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
targets = data['targets']
train_x = data['train_x']
train_y = data['train_y']


# import our chat-bot samples file
import json
with open('samples.json') as json_data:
    samples = json.load(json_data)


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# load our saved model
model.load('./model.tflearn')

#--------------------------------------------------------------#

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def BagOfWords(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))


#----------------------------------------------------------------#

# create a data structure to hold user context which is data 
# realted to previous question asked by the same user
context = {}
ERROR_THRESHOLD = 0.25

def classify(sentence):
    # generate probabilities from the model
    results = model.predict([BagOfWords(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((targets[r[0]], r[1]))
    # return tuple of sample and probability
    return return_list

# sserID is a sqme persons conversatioins 
def answer(sentence, userID='33', show_details=True):
    results = classify(sentence)
    # if we have a classification then find the matching sample tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in samples['samples']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this sample if necessary
                    if 'context_set' in i:
                        if show_details:

                         	print('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # Checking if this sample is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random answer from the sample
                        return random.choice(i['answers'])

            results.pop(0)



def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        print(answer(sentence))
        
chat()    




