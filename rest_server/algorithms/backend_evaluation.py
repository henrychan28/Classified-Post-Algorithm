import pickle
from .group_classification import topic_classification
import os
import sys

# list_of_good_words is a list of list.
# list_of_good_words[i][j] returns the word with j-th highest score in group i
def initialization():
    current_directory = os.getcwd() + '\\algorithms'
    global list_of_good_words
    list_of_good_words = pickle.load(open(current_directory + '\\list_of_good_words.pickle', "rb"))

    # word_scores is a list of dict
    # word_scores[i]['wwww'] is the score of the word 'wwww' in group i
    global word_scores
    word_scores = pickle.load(open(current_directory+"\\word_scores.pickle", "rb" ))

    # max_scores is a list
    # max_scores[i] is the max score of words stored in group i
    global max_scores
    max_scores = pickle.load(open(current_directory+"\\max_scores.pickle", "rb"))
    global ignore_list
    ignore_list = ['hong', 'kong']


####### copy from stem_stop_group.py
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk


def stem_and_stop(raw):
    tokens = word_tokenize(raw)

    wnl = nltk.WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    stem_stop_words = []

    for t in tokens:
        lemmatized = wnl.lemmatize(t.lower())
        if (lemmatized not in stopWords) and (len(lemmatized) > 3) and (lemmatized not in ignore_list):
            stem_stop_words.append(lemmatized)

    return(stem_stop_words)
###### end of copy


def good_words(topic):
    # TODO: call Henry function to determine group
    try:
        group = topic_classification(topic)
    except KeyError:
        print("No Such Key")
        return ["ERROR: Invalid topic - Please check your spelling"]
    return list_of_good_words[group]

# score is between 0 and 10
def score(title, topic):
    # TODO: call Henry function to determine Group
    try:
        group = topic_classification(topic)
    except KeyError:
        print("No Such Key")
        return -1
    count = 0
    total_score = 0
    for word in stem_and_stop(title):
        if word in word_scores[group]:
            count += 1
            total_score += word_scores[group][word]
    if(count > 0):
        return total_score / count / max_scores[group] * 10
    else:
        return 0 # all words in title do not in training data for that group



# test
#print(good_words("technology"))
#print(score("Alibaba and Tencent went bankrupt", "technology"))
