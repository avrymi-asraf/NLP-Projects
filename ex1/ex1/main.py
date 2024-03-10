import spacy
from datasets import load_dataset
from collections import Counter
import operator
import pickle
import math

START_TOKEN = "STARTSTART"


def load_corpus(corpus_name):
    nlp = spacy.load(corpus_name)  # en_core_web_sm
    nlp.max_length = 11106903 + 10
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    raw_text = START_TOKEN + "".join(text['text'])
    raw_text = raw_text.replace("\n", START_TOKEN)
    doc = nlp(raw_text)

    # filter the doc for only alphabetic word and convert to lemma
    list_doc = list()
    for t in doc:
        if t.is_alpha:
            list_doc.append(t.lemma_)  # list of doc

    return list_doc


def train_unigram(nlp_list):
    num_word_on_corpus = len(nlp_list)
    unigram_model = Counter(nlp_list)
    for key in unigram_model:
        unigram_model[key] /= num_word_on_corpus

    return dict(unigram_model)


def unigram_predict(nlp_dict):

    probable_word = max(nlp_dict.items(), key=operator.itemgetter(1))[0]
    return probable_word


def train_bigram(nlp_list):
    bigram_model = {key: {} for key in nlp_list}

    # counting the number of appearances of each bigram
    for i in range(len(nlp_list) - 1):
        current_word = nlp_list[i]
        next_word = nlp_list[i + 1]

        if next_word == START_TOKEN:  # end of sentence
            continue

        sub_dict = bigram_model[current_word]

        if next_word in sub_dict:
            sub_dict[next_word] += 1
        else:
            sub_dict[next_word] = 1

    for curr_word_key in bigram_model:
        sub_dict = bigram_model[curr_word_key]
        sum_sub_dict = sum(sub_dict.values())
        for nex_word_key in sub_dict:
            sub_dict[nex_word_key] /= sum_sub_dict

    return bigram_model


def bigram_predict(nlp_dict, current_word):

    sub_dict = nlp_dict[current_word]
    probable_word = max(sub_dict.items(), key=operator.itemgetter(1))[0]
    return probable_word


def bigram_log_probability(nlp_dict, sentence):
    nlp = spacy.load("en_core_web_sm")  # en_core_web_sm
    doc = nlp(sentence)
    list_sentence = list()
    list_sentence.append(START_TOKEN)
    for t in doc:
        if t.is_alpha:
            list_sentence.append(t.lemma_)  # list of doc

    res = 0

    for i in range(len(list_sentence) - 1):
        current_word = list_sentence[i]
        next_word = list_sentence[i + 1]

        if current_word in nlp_dict:
            sub_dict = nlp_dict[current_word]
            if next_word in sub_dict:
                res += math.log(sub_dict[next_word])
            else:
                # print('\t The word "', next_word, '" does not exist in corpus after "', current_word, '"')
                return float('-inf')
        else:
            # print('\t The word "', current_word, '" does not exist in corpus')
            return float('-inf')

    return res


def backoff_log_probability(bi_model, uni_model, lambda_bi, lambda_uni, sentence):
    nlp = spacy.load("en_core_web_sm")  # en_core_web_sm
    doc = nlp(sentence)
    list_sentence = list()
    list_sentence.append(START_TOKEN)
    for t in doc:
        if t.is_alpha:
            list_sentence.append(t.lemma_)  # list of sentence

    res = 0

    for i in range(len(list_sentence) - 1):
        current_word = list_sentence[i]
        next_word = list_sentence[i + 1]

        if current_word in bi_model:
            sub_bi_model = bi_model[current_word]
            if next_word in sub_bi_model:
                res += math.log(lambda_bi*sub_bi_model[next_word] + lambda_uni*uni_model[next_word])
            else:  # sub_bi_model[next_word] = 0
                res += math.log(lambda_uni*uni_model[next_word])
        else:
            # print('\t The word "', current_word, '" does not exist in corpus')
            return float('-inf')

    return res


def main():

    nlp_list = load_corpus("en_core_web_sm")

    trained_bigram_model = train_bigram(nlp_list)
    trained_unigram_model = train_unigram(nlp_list)

#############################################################################################
    print("2) bigram auto complete: ")
    print("\t I have a house in", bigram_predict(trained_bigram_model, "in"), "\n")

#############################################################################################
    print("3) bigram probability and perplexity: ")

    first_sentence = "Brad Pitt was born in Oklahoma"
    res = bigram_log_probability(trained_bigram_model, first_sentence)
    print('\t Log probability of "', first_sentence, '" is: ', res, "\n")

    second_sentence = "The actor was born in USA"
    res = bigram_log_probability(trained_bigram_model, second_sentence)
    print('\t Log probability of "', second_sentence, '" is: ', res, "\n")

    count_of_words = len((first_sentence + " " + second_sentence).split())
    l = 1/count_of_words*(bigram_log_probability(trained_bigram_model, first_sentence) + bigram_log_probability(trained_bigram_model, second_sentence))
    print('\t The perplexity is: ', math.exp(-l), "\n")
###################################################################################################

    print("4) Back-Off probability and perplexity: ")

    lambda_uni = 1 / 3
    lambda_bi = 2 / 3

    res = backoff_log_probability(trained_bigram_model, trained_unigram_model, lambda_bi, lambda_uni, first_sentence)
    print('\t Log probability of "', first_sentence, '" is: ', res, "\n")

    res = backoff_log_probability(trained_bigram_model, trained_unigram_model, lambda_bi, lambda_uni, second_sentence)
    print('\t Log probability of "', second_sentence, '" is: ', res, "\n")

    l = 1 / count_of_words * (backoff_log_probability(trained_bigram_model, trained_unigram_model, lambda_bi, lambda_uni, first_sentence) + backoff_log_probability(trained_bigram_model, trained_unigram_model, lambda_bi, lambda_uni, second_sentence))
    print('\t The perplexity is: ', math.exp(-l), "\n")


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
