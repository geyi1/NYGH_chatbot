import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

if __name__ == '__main__':
    output = {}

    f = open(r'C:\Users\Admin\Desktop\NYGH.json')
    data = json.load(f)["intents"]
    output = {}
    output['intents'] = []
    count = 0
    aug = nac.RandomCharAug(aug_char_p=0.1)
    aug2 = naw.SynonymAug(aug_p=0.1, stopwords=["how", "i", "I", "covid", "covid-19", "Covid", "Covid-19", "it", "Canada", "Ontario", "canada", "ontario"])
    aug3 = nac.RandomCharAug(aug_char_p=0.1, action="delete")
    aug4 = naw.ContextualWordEmbsAug(action="insert", aug_p=0.1, stopwords="QR")
    aug5 = naw.ContextualWordEmbsAug(action="substitute", aug_p=0.1, stopwords="QR")
    back_trans_aug = naw.BackTranslationAug()

    aug_list = [aug, aug2, aug3, aug4, aug5]

    replace_list = {"dose": ["shot"], "doses": ["shots"], "vaccine": ["vaccination", "shot", "immunized"], "covid": ["covid-19", "coronavirus", "virus"], "booster": ["booster shot", "third dose"]}

    for block in data:
        print("current tag: ", block["tag"])

        augmented_data = {}
        augmented_data["tag"] = block["tag"]
        augmented_data["responses"] = block["responses"]
        augmented_data["patterns"] = []

        temp = []

        for each in block["patterns"]:
            temp.append(each)
            back_trans_str = back_trans_aug.augment(each)
            if back_trans_str != each:
                temp.append(back_trans_str)
            for word in replace_list:
                if word in each:
                    for replace_word in replace_list[word]:
                        if replace_word not in each:
                            augmented_text = each.replace(word, replace_word)
                            temp.append(augmented_text)

        for each in temp:
            count = 0
            for aug in aug_list:
                # print("count is ",count)
                # count += 1
                augmented_text = aug.augment(each, n=2)
                for text in augmented_text:
                    augmented_data['patterns'].append(text)

        output['intents'].append(augmented_data)

    with open("augmented_data.json", 'w') as file:
        json.dump(output, file)


