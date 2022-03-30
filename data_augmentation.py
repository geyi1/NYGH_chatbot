import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from preprocessing import helper

if __name__ == '__main__':
    output = {}

    f = open(r'C:\Users\Admin\Desktop\NYGH.json')
    data = json.load(f)["intents"]
    output = {}
    output['intents'] = []
    count = 0
    aug = nac.RandomCharAug(aug_char_p=0.2)
    aug2 = naw.SynonymAug(aug_p=0.2, stopwords=["how", "i", "I", "covid", "covid-19", "Covid", "Covid-19", "it", "Canada", "Ontario", "canada", "ontario"])
    aug3 = nac.RandomCharAug(aug_char_p=0.2, action="delete")
    aug4 = naw.ContextualWordEmbsAug(action="insert", aug_p=0.2, stopwords="QR")
    aug5 = naw.ContextualWordEmbsAug(action="substitute", aug_p=0.2, stopwords="QR")
    back_trans_aug = naw.BackTranslationAug()

    aug_list = [aug, aug2, aug3, aug4, aug5]

    replace_list = {
                    "dose": ["shot"], "doses": ["shots"], "vaccine": ["vaccination", "vaccine shot", "shot", "immunized"],
                    "covid": ["covid-19", "coronavirus", "virus"], "booster": ["booster shot", "third dose"],
                    "vaccination": ["vaccine", "shot", "immunized"], "receipt": ["confirmation"],
                    "confirmation": ["receipt"], "fever": ["tiredness", "muscle pain", "chills", "cough", "chest pain", "coughing"],
                    "can": ["how do", "do", "could", "would", "how could", "how would", ], "make": ["get", "book"], "get": ["make", "book"],
                    "book": ["make", "get"], "severe": ["extreme", "really bad", "worst"],
                    "i": ["my child", "a child", "my parent", "my dad", "my mum", "my dad", "my mum", "my year old"],
                    }

    for block in data:
        print("current tag: ", block["tag"])

        augmented_data = {}
        augmented_data["tag"] = block["tag"]
        augmented_data["responses"] = block["responses"]
        augmented_data["patterns"] = []

        temp = []

        for each in block["patterns"]:
            each = helper.preprocess(each)
            print("the pattern is: ")
            print(each)
            temp.append(each)
            split_word = each.split()
            if block["tag"] != "greetings" and block["tag"] != "thankyou":
                back_trans_str = back_trans_aug.augment(each)
                if back_trans_str != each and back_trans_str and abs(len(each) - len(back_trans_str))/len(each) <= 1:
                    back_trans_str = helper.preprocess(back_trans_str)
                    print("back trans is ")
                    print(back_trans_str)
                    temp.append(back_trans_str)
            for replace_word in replace_list:
                if replace_word in split_word:
                    if block["tag"] == "vaccineAppointment":
                        if replace_word == "vaccine":
                            augmented_text = ''
                            for word in split_word:
                                if word == replace_word:
                                    new_word = "booster"
                                else:
                                    new_word = word
                                augmented_text += new_word
                                augmented_text += ' '
                            augmented_text = augmented_text.rstrip()
                            temp.append(augmented_text)
                    if block["tag"] != "greetings" and block["tag"] != "thankyou":
                        for replacement in replace_list[replace_word]:
                            if replacement not in split_word:
                                augmented_text = ''
                                for word in split_word:
                                    if replace_word == word:
                                        new_word = replacement
                                    else:
                                        new_word = word
                                    augmented_text += new_word
                                    augmented_text += ' '
                                augmented_text = augmented_text.rstrip()
                                temp.append(augmented_text)



        for each in temp:
            count = 0
            for aug in aug_list:
                # print("count is ",count)
                # count += 1
                augmented_text = aug.augment(each, n=3)
                print("augmented text is")
                print(augmented_text)
                for text in augmented_text:
                    augmented_data['patterns'].append(text)

        output['intents'].append(augmented_data)


    with open("augmented_data.json", 'w') as file:
        json.dump(output, file)


