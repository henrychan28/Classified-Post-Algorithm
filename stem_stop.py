from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk

import pickle
import operator

def stem_and_stop(raw):
    tokens = word_tokenize(raw)

    wnl = nltk.WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    stem_stop_words = []

    for t in tokens:
        lemmatized = wnl.lemmatize(t.lower())
        if lemmatized not in stopWords and len(lemmatized) > 3:
            stem_stop_words.append(lemmatized)

    return(stem_stop_words)

with open('data/data_with_topics_and_group.pickle', 'rb') as data_file:
    data = pickle.load(data_file)

# use first 10 elements to test
data = data[0:10]
print(data)
# print('=======')

#print(data)


document_frequencies = dict()
final_weights = dict()

# populate document_frequencies
for entry in data:
#    raw = entry['Title'] + ' ' + entry['Descriptions']
    raw = entry['Title']
    # remove duplicate in cleaned data
    cleaned = set(stem_and_stop(raw))
    for word in cleaned:
        if word in document_frequencies:
            document_frequencies[word] += 1
        else:
            document_frequencies[word] = 1

# calculate final_weights
for entry in data:
#    raw = entry['Title'] + ' ' + entry['Descriptions']
    raw = entry['Title']
    page_views = int(entry['Pageviews'].replace(',', ''))
    tf_idf = dict()
    '''
    for word in stem_and_stop(raw):
        if word in tf_idf:
            tf_idf[word] += 1/document_frequencies[word]
        else:
            tf_idf[word] = 1/document_frequencies[word]
    '''
    for word in stem_and_stop(raw):
            # tf_idf[word] = 1/document_frequencies[word]
            tf_idf[word] = 1

    # total_tf_idf = sum(tf_idf.values())

    # calculate the final weights
    for word in stem_and_stop(raw):
        #weight = tf_idf[word] / total_tf_idf * page_views
        weight = tf_idf[word] * page_views
        if word in final_weights:
            final_weights[word] += weight
        else:
            final_weights[word] = weight

sorted_final_weights = sorted(final_weights.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_final_weights[0:10])
