from string import punctuation
import re

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
        # if training:
        #     temp = sentence.split()
        #     sentence = ' '.join([word for word in temp if not word.isdigit()])
        return sentence