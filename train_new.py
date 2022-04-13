import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.optimizers import SGD
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import random
from preprocessing import helper
from matplotlib import pyplot as plt
#
# path = r'C:\Users\Admin\Desktop\NYGH.json'

# if __name__ == '__main__':

'''
this python file: 
- trains a model with datas in augmented_data.json
- save the training and validation accuracy as pdfs
- save model for further usage
'''

lemmatizer = WordNetLemmatizer()
with open(r'augmented_data.json') as f:
    data = json.load(f)

words = []
classes = []
documents = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        clean_pattern = helper.preprocess(pattern)
        w = nltk.word_tokenize((clean_pattern))
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# print(words)
# print(classes)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
# print(train_x)
# print(train_y)
# print(len(words))
# print(len(train_x[0]))
# print("Training data created")
#
#
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model
trained_model = model.fit(np.array(train_x), np.array(train_y), epochs=30, batch_size=32, verbose=1, validation_split=0.1)
print(trained_model.history)
plt.plot(trained_model.history['accuracy'])
plt.plot(trained_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('training_accuracy_3.pdf')
plt.show()

plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('training_loss_3.pdf')
plt.show()
model.save('chatbot_model_3.h5', trained_model)
print("model created")

