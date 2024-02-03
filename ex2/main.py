#import torch
import numpy as np
import nltk as nl
from nltk.corpus import brown
from collections import Counter

## run it once on first time
#nl.download('averaged_perceptron_tagger')
#nl.download('punkt')
START_TOKEN = 'STARTSTART'
start_token_tagged = nl.pos_tag(nl.word_tokenize(START_TOKEN))
WORD_POSITION = 0
TAG_POSITION = 1
UNKNOWN_WORD_TAG = 'NN'
def contains_only_nonalphabetic(input_string):
    return not any(char.isalpha() for char in input_string)


def cut_until_nonalphabetic(input_string):
    result = ""
    for char in input_string:
        if not char.isalpha():
            break
        result += char
    return result


def tag_filter(tag):
    if tag.isalpha() or contains_only_nonalphabetic(tag):
        return tag
    else:
        return cut_until_nonalphabetic(tag)

# This function imports the database, adds the start token and make it list of words (and not sentences)
def import_db():
    brown_db = brown.tagged_sents(categories='news')
    # add start token for beginning of sentence
    brown_db_with_start = [start_token_tagged + sentence for sentence in brown_db]

    # make training set
    training_set = brown_db_with_start[:round(0.9 * len(brown_db_with_start))]
    training_set_flat = [item for sublist in training_set for item in sublist]
    training_set_with_good_tags = [(word[0], tag_filter(word[1])) for word in training_set_flat]

    # make test set
    test_set = brown_db_with_start[round(0.9 * len(brown_db_with_start)):]
    test_set_flat = [item for sublist in test_set for item in sublist]
    test_set_with_good_tags = [(word[0], tag_filter(word[1])) for word in test_set_flat]

    return [training_set_with_good_tags, test_set_with_good_tags]

def most_likely_baseline(training_set, tagged_word):

    training_set_quantities = Counter(training_set)

    max_of_occur = 0
    tag_prediction = UNKNOWN_WORD_TAG

    for word in training_set:
        if word[WORD_POSITION] == tagged_word[WORD_POSITION] and training_set_quantities[word] > max_of_occur:
            max_of_occur = training_set_quantities[word]
            tag_prediction = word[TAG_POSITION]

    return tag_prediction

def word_in_training_set_checker(training_set, tagged_word):
    for word in training_set:
        if word[WORD_POSITION] == tagged_word[WORD_POSITION]:
            return True
    return False

def main():
    [training_set, test_set] = import_db()

    # b section - DONE
    test_set_unique_words = list(set(tuple(t) for t in test_set))
    test_set_unique_words.remove(start_token_tagged[0])
    num_of_words = len(test_set_unique_words)
    i = 0
    correct_tags_known_words = 0
    amount_known_words = 0
    correct_tags_unknown_words = 0
    amount_unknown_words = 0
    for word in test_set_unique_words:
        i += 1
        if word_in_training_set_checker(training_set, word):
            if word[TAG_POSITION] == most_likely_baseline(training_set, word):
                correct_tags_known_words += 1
            amount_known_words += 1
        else:
            if word[TAG_POSITION] == most_likely_baseline(training_set, word):
                correct_tags_unknown_words += 1
            amount_unknown_words += 1

    print("The error rate for known words is   ", 1-correct_tags_known_words/amount_known_words)
    print("The error rate for unknown words is ", 1-correct_tags_unknown_words/amount_unknown_words)
    print("The total error rate is ", 1-(correct_tags_known_words + correct_tags_unknown_words)/(amount_unknown_words + amount_known_words))
    pass


if __name__ == '__main__':
    main()
