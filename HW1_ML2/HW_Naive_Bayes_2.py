import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter



df = pd.read_csv("Data/Data.txt", sep="\t")
df.columns = ["status", "body"]
data_train, data_test = train_test_split(df, test_size=0.20, random_state=16)

spam_messages = df.loc[df["status"] == 'spam']
ham_messages = df.loc[df["status"] == 'ham']

def create_dict(data):
    words = [(word.lower()) for word in data["body"].str.split().sum() if word.isalpha() and len(word) > 1]
    count = Counter(words)
    dict = pd.DataFrame.from_dict(count, orient="index")
    dict = dict.rename(columns={'index': 'word', 0:'count'})
    print(dict)
    dict['count'] = dict['count']
    return len(words), dict

def naive_bayes(email, spam_dict, ham_dict, nbr_spam_word, nbr_ham_word):
    words = email.lower().split(" ")
    Pspam = len(spam_messages)/(len(spam_messages)+len(ham_messages))
    Pham = 1-Pspam
    Pword = 1.
    Pword_w_spam = 1.

    for word in words:
        if word in ham_dict.index or word in spam_dict.index:
            nbr_spam = (spam_dict._get_value(word, 'count')+1 if word in spam_dict.index else 1)/(nbr_spam_word+1)
            nbr_norm = (ham_dict._get_value(word, 'count')+1 if word in ham_dict.index else 1)/(nbr_ham_word+1)
            Pword = Pword * nbr_norm
            Pword_w_spam = Pword_w_spam * nbr_spam
    Pspam_w_email = Pspam * Pword_w_spam
    return Pspam_w_email, Pword * Pham


spam_words_nbr, spam_dict = create_dict(spam_messages)
ham_words_nbr, ham_dict = create_dict(ham_messages)

faux_pos = faux_neg = vrai_pos = vrai_neg = 0


test_result = np.zeros((len(data_test), 3))
index = 0

#Pspam, Pham = naive_bayes('Do have a nice day today. I love you so dearly.', spam_dict, ham_dict, spam_words_nbr, ham_words_nbr)
test = pd.DataFrame(data_test).to_numpy()
for message in test:
    print(message[1])
    Pspam, Pham = naive_bayes(message[1], spam_dict, ham_dict, spam_words_nbr, ham_words_nbr)
    print(Pspam)
    test_result[index] = [Pspam, Pham,(0) if message[0] == 'ham' else 1]
    print(test_result[index])
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

