
# Installing Dependencies for  NLP
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random


# Loading our chatbot samples data file
import json
with open('samples.json') as json_data:
    samples = json.load(json_data)


words = []
targets = []
documents = []
ignore_words = ['?']
# loop through each sentence in our samples questions
for sample in samples['samples']:
    for question in sample['questions']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(question)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, sample['tag']))
        # add to our targets list
        if sample['tag'] not in targets:
            targets.append(sample['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
targets = sorted(list(set(targets)))

#print (len(documents), "documents")
#print (len(targets), "targets", targets)
#print (len(words), "unique stemmed words", words)

#-------------------------------------------------------------------------------------#


# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(targets)

# training set, bag of words for each sentence
for doc in documents:
    
    # initialize our bag of words
    bag = []
    # list of tokenized words for the question
    question_words = doc[0]
    # stem each word
    question_words = [stemmer.stem(word.lower()) for word in question_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in question_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[targets.index(doc[1])] = 1

    training.append([bag, output_row])

# Lets rearrange our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test data 
train_x = list(training[:,0])
train_y = list(training[:,1])


#-------------------------------------------------------------------------------------#

# Reset graph 
tf.reset_default_graph()
# Build neural network
# used Softmax activation functions as there will be multiple classes
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Training model using gradient descent algorithm and Saving it.
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# Saving our data structures
import pickle
pickle.dump( {'words':words, 'targets':targets, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )



