import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random
import re
from string import punctuation

intents = json.loads(open(r'augmented_data.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


class helper:
    @staticmethod
    def remove_double_spaces(str):
        return " ".join(str.split())

    @staticmethod
    def remove_punctuation(str):
        return ''.join(c for c in str if c not in punctuation)

    @staticmethod
    def decontractions(phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"won\’t", "will not", phrase)
        phrase = re.sub(r"can\’t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        phrase = re.sub(r"n\’t", " not", phrase)
        phrase = re.sub(r"\’re", " are", phrase)
        phrase = re.sub(r"\’s", " is", phrase)
        phrase = re.sub(r"\’d", " would", phrase)
        phrase = re.sub(r"\’ll", " will", phrase)
        phrase = re.sub(r"\’t", " not", phrase)
        phrase = re.sub(r"\’ve", " have", phrase)
        phrase = re.sub(r"\’m", " am", phrase)
        return phrase

    @staticmethod
    def preprocess(sentence):
        sentence = sentence.lower()
        sentence = helper.remove_punctuation(sentence)
        sentence = helper.remove_double_spaces(sentence)
        sentence = helper.remove_punctuation(sentence)
        return sentence


def process_input(sentence, words):
    # tokenize the pattern
    sentence = helper.preprocess(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = process_input(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res)]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def print_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            print("class is: ", tag)
            result = random.choice(i['responses'])
            break
    return result


def predict_response(text):
    ints = predict_class(text, model)
    res = print_response(ints, intents)
    # print(ints)
    print("********************************")
    return res


# while True:
#     user = input("give me a string: ")
#     print("response is: ", predict(user))
#     print("*****************")

# test_dict = {"vaccineAppointment": ["how do I get a vaccine appointment", "how do I book a vaccine appointment", "where do I book a vaccine appointment", "how to book vaccine appointment", "can I rebook appointment", "how to cancel appointment"],
#              "vaccineConfirmation": ["how do I get a receipt for my vaccine", "how will i get confirmation for vaccine", "will i get receipt for vaccination", "how to get pdf for vaccine", "where do i get vaccine confirmation"],
#              "covidEmergency": ["I am having serious fever, what should I do", "I am having severe breathing problem", "I have chest pain", "I am immunocompromise and got covid"],
#              "covidGeneral": ["When will covid end", "What are the side effects of vaccine", "what are the side effects of covid-19", "What is the long term effect of covid", "why do I need vaccine", "what medication should I take for covid at home"],
#              "covidQRCode": ["How do I get QR code for my vaccine", "Does BC qr code work", "I dont have a health card, how do I get a qr code", "I got vaccine outside of canada, how do i get a qr code", "where is qr code required"],
#              "mixingVaccine": ["can i mix vaccine", "Is it safe to mix vaccines", "what are the effects of mixing vaccine", "do i have to mix vaccine", "why should i mix vaccine"],
#              "covidTravel": ["Can i go to other countries if I am vaccinated", "where do i get tested for covid", "what kind of test do i need to travel", "how long before should i get tested", "do i still need to quarantine for travelling"],
#              "covidBooster": ["Why do i need a booster shot", "is booster the same as other doses", "can i get a different vaccine for booster", "side effect of booster", "how long should i wait to get booster", "do i need to get booster if i already got covid"]}
#
# count = 0
# success = 0
# label_count = 0
# label_success = 0
# for label in test_dict:
#     label_count = 0
#     label_success = 0
#     for each in test_dict[label]:
#         temp = predict_class(each, model)[0]['intent']
#         if label == temp:
#             success += 1
#             label_success += 1
#         count += 1
#         label_count += 1
#     print("label is {}, accuracy is {}".format(label, label_success / label_count * 100))
#     print("***********************************************")
# print("overall accuracy is ", success / count * 100)

