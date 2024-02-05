#import torch
import numpy as np
import nltk as nl
from nltk.corpus import brown
from collections import Counter

## run it once on first time
#nl.download('averaged_perceptron_tagger')
#nl.download('punkt')
START_TOKEN = 'STARTSTART'
START_TAG = 'STARTTAG'
#start_token_tagged = nl.pos_tag(nl.word_tokenize(START_TOKEN))
start_token_tagged = [(START_TOKEN, START_TAG)]
WORD_POSITION = 0
TAG_POSITION = 1
UNKNOWN_WORD_TAG = 'NN'
SECTION = 'c'
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


def train_transition_hmm(training_set):
    hmm = {key[TAG_POSITION]: {} for key in training_set}

    # counting the number of appearances of each bigram
    for i in range(len(training_set) - 1):
        current_word = training_set[i]
        next_word = training_set[i + 1]

        if next_word[WORD_POSITION] == START_TOKEN:  # end of sentence
            continue

        sub_dict = hmm[current_word[TAG_POSITION]]

        if next_word[TAG_POSITION] in sub_dict:
            sub_dict[next_word[TAG_POSITION]] += 1
        else:
            sub_dict[next_word[TAG_POSITION]] = 1

    for curr_tag_key in hmm:
        sub_dict = hmm[curr_tag_key]
        sum_sub_dict = sum(sub_dict.values())
        for nex_tag_key in sub_dict:
            sub_dict[nex_tag_key] /= sum_sub_dict

    return hmm



def train_emission_hmm(training_set):
    tags_possible = {key[TAG_POSITION]: 0 for key in training_set}
    hmm = {key[WORD_POSITION]: tags_possible for key in training_set}
    training_set_tags = [tagged_word[TAG_POSITION] for tagged_word in training_set]
    training_set_tags_with_amount = Counter(training_set_tags)

    for tagged_word in training_set:
        hmm[tagged_word[WORD_POSITION]] = hmm[tagged_word[WORD_POSITION]].copy()
        hmm[tagged_word[WORD_POSITION]][tagged_word[TAG_POSITION]] += 1

    for word_emission in hmm.values():
        for tag in word_emission:
            word_emission[tag] /= training_set_tags_with_amount[tag]

    return hmm




def main():
    [training_set, test_set] = import_db()

    if SECTION == 'b':
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

    # c section
    elif SECTION == 'c':
        #transition_hmm = train_transition_hmm(training_set)
        emission_hmm = train_emission_hmm((training_set))
        pass













if __name__ == '__main__':
    main()
