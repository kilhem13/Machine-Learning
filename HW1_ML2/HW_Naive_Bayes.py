import os
import numpy as np
from collections import Counter
Dict = []
X = []
Y = []
def parsing_data(filename):
    with open(os.path.join(os.getcwd()+"/Data", filename), 'r') as f:
        for line in f.readlines():
           [Dict.append(word) for word in line.split("\t")[1].split(" ")]
        dictionnaire = Counter(Dict)
        for word in list(dictionnaire):
            if word in dictionnaire:
                if word.isalpha() == False or len(word) == 1:
                    del dictionnaire[word]
        dictionnaire = dictionnaire.most_common(3000)
        f.close()
        return dictionnaire

def extract_features(filename, dictionnaire):
    with open(os.path.join(os.getcwd()+"/Data", filename), 'r') as f:
        features_matrix = np.zeros((len(f.readlines()), 3000))
        f.seek(0)
        lineID = 0
        for line in sorted(f.readlines()):
            words = line.split()
            for word in words:
                word = word.lower()
                for i, w in enumerate(dictionnaire):
                    if w[0] == word:
                        wordID = i
                        features_matrix[lineID, wordID] = words.count(word)
            lineID = lineID + 1
    f.close()
    return features_matrix


def get_label(filename):
    with open(os.path.join(os.getcwd() + "/Data", "training_data.txt"), 'r') as f:
        lines = f.readlines()
        sorted_labels = sorted((line.split("\n")[0].split("\t")[0]) for line in lines)
        f.seek(0)
        train_label = np.zeros(len(f.readlines()))
        index = 0
        for label in sorted_labels:
            if label == 'ham':
                index = index + 1
            else:
                break
        train_label[index::] = 1
    return train_label, index

def get_count(filename, spam_index):
    Dict_spam = []
    Dict_norm = []
    with open(os.path.join(os.getcwd()+"/Data", filename), 'r') as f:
        i = 0
        for line in sorted(f.readlines()):
            if i > spam_index:
                [Dict_spam.append(word) for word in line.split("\t")[1].split(" ")]
            else:
                [Dict_norm.append(word) for word in line.split("\t")[1].split(" ")]
            i = i+1
        spam_dictionnaire = Counter(Dict_spam)
        norm_dictionnaire = Counter(Dict_norm)
        for word in list(spam_dictionnaire):
            if word in spam_dictionnaire:
                if word.isalpha() == False or len(word) == 1:
                    del spam_dictionnaire[word]
        spam_dictionnaire = spam_dictionnaire.most_common(3000)
        for word in list(norm_dictionnaire):
            if word in norm_dictionnaire:
                if word.isalpha() == False or len(word) == 1:
                    del norm_dictionnaire[word]
        norm_dictionnaire = norm_dictionnaire.most_common(3000)
        f.close()
        return spam_dictionnaire, norm_dictionnaire


def naive_bayes(train_label, last_ham_index, email, spam_count, dictionnaire):
    email = email.lower()
    words = email.split(" ")
    Pspam = (len(train_label)-last_ham_index)/len(train_label)
    Pham = 1-Pspam
    Pword = 1.
    Pword_w_spam = 1.
    nbr_spam_word = 0
    nbr_ham_word = 0
    for word in spam_count:
        nbr_spam_word += word[1]
    for word in dictionnaire:
        nbr_ham_word += word[1]

    for word in words:
        if dict(dictionnaire).get(word) or dict(spam_count).get(word):
            nbr_spam = (dict(spam_count).get(word)+1 if dict(spam_count).get(word) else 1)/nbr_spam_word
            nbr_norm = (dict(dictionnaire).get(word)+1 if dict(dictionnaire).get(word) else 1)/nbr_ham_word
            Pword = Pword * nbr_norm
            Pword_w_spam = Pword_w_spam * nbr_spam
    Pspam_w_email = Pspam * Pword_w_spam
    return Pspam_w_email, Pword * Pham


dictionnaire = parsing_data("training_data.txt")
features_matrix = extract_features("training_data.txt", dictionnaire)
train_label, last_ham_index = get_label("training_data.txt")
spam_count, norm_count = get_count("training_data.txt", last_ham_index)
email = "FREE for 1st week! No1 Nokia tone 4 ur mob every week just txt NOKIA to 8007 Get txting and tell ur mates www.getzed.co.uk POBox 36504 W45WQ norm150p/tone 16+"
Pspam, Pham = naive_bayes(train_label, last_ham_index, email, spam_count, norm_count)

test_label, test_index = get_label("test_data.txt")
faux_pos = 0
faux_neg = 0
vrai_pos = 0
vrai_neg = 0
with open(os.path.join(os.getcwd() + "/Data", "test_data.txt"), 'r') as f:
    lines = f.readlines()
    sorted_messages = sorted((line.split("\t")) for line in lines)
    f.seek(0)
    test_result = np.zeros((len(f.readlines()),3))
    index = 0
    for message in sorted_messages:
        Pspam, Pham = naive_bayes(train_label, last_ham_index, message[1].split("\n")[0], spam_count, norm_count)
        test_result[index] = [Pspam, Pham,(0) if message[0] == 'ham' else 1]
        if Pspam > Pham and test_result[index][2] == 0:
            faux_pos = faux_pos +1
        elif Pspam > Pham and test_result[index][2] > 0:
            vrai_pos = vrai_pos +1
        elif Pspam < Pham and test_result[index][2] == 0:
            vrai_neg = vrai_neg +1
        else:
            faux_neg = faux_neg +1
        index = index + 1
print("Vrais Positifs: ", vrai_pos, "\nFaux Positifs: ", faux_pos, "\nVrais Negatifs: ", vrai_neg, "\nFaux Negatifs: ", faux_neg)





