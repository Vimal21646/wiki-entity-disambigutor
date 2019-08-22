from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from itertools import product

import numpy

# Defining stopwords for English Language
stop_words = set(stopwords.words("english"))


# Method  to tokenizing and removing the Stopwords
def tokenize_remove_stop_words(sentence):
    filtered_sentence = []
    for words in word_tokenize(sentence):
        if words not in stop_words:
            if words.isalnum():
                filtered_sentence.append(words)
    return filtered_sentence


# Method for lemmatizing: Root Words
def lemmatize_sentence(filtered_sentence):
    lemm_sentence = []

    # Defining WordNet Lematizer for English Language
    lemmatizer = WordNetLemmatizer()

    for i in filtered_sentence:
        lemm_sentence.append(lemmatizer.lemmatize(i))
    return lemm_sentence


def cal_sentence_similarity(str1, str2):
    # Initialising List
    similarity_probs = []

    # Tokenizing and removing the Stopwords
    filtered_sentence1 = tokenize_remove_stop_words(str1)

    # Lemmatizing: Root Words
    lemm_sentence1 = lemmatize_sentence(filtered_sentence1)

    # Tokenizing and removing the Stopwords
    filtered_sentence2 = tokenize_remove_stop_words(str2)

    # Lemmatizing: Root Words
    lemm_sentence2 = lemmatize_sentence(filtered_sentence2)

    # Similarity index calculation for each word
    for word1 in lemm_sentence1:
        simi = []
        for word2 in lemm_sentence2:
            sims = []
            syns1 = wordnet.synsets(word1)
            syns2 = wordnet.synsets(word2)
            for sense1, sense2 in product(syns1, syns2):
                d = wordnet.wup_similarity(sense1, sense2)
                if d is not None:
                    sims.append(d)

            if sims:
                max_sim = max(sims)
                simi.append(max_sim)

        if simi:
            max_final = max(simi)
            similarity_probs.append(max_final)

    # Final Output

    similarity_index = numpy.mean(similarity_probs)
    similarity_index = round(similarity_index, 2)
    return similarity_index
