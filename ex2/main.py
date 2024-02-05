# import torch
import numpy as np
import nltk as nl
from nltk.corpus import brown
from collections import Counter

## run it once on first time
# nl.download('averaged_perceptron_tagger')
# nl.download('punkt')
START_TOKEN = "STARTSTART"
START_TAG = "STARTTAG"
UNKNOWN_WORD = "UNKNOWNUNKNOWN"
UNKNOWN_WORD_TAG = "NN"
# start_token_tagged = nl.pos_tag(nl.word_tokenize(START_TOKEN))
start_token_tagged = [(START_TOKEN, START_TAG)]
WORD_POSITION = 0
TAG_POSITION = 1
SECTION = "c"


def contains_only_nonalphabetic(input_string: str) -> bool:
    """check if the input string contains only non-alphabetic characters

    Args:
        input_string (str): text to be checked

    Returns:
        bool: True if the input string contains only non-alphabetic characters, False otherwise
    """
    return not any(char.isalpha() for char in input_string)


def cut_until_nonalphabetic(input_string: str) -> str:
    """return the prefix of the input string until the first non-alphabetic character

    Args:
        input_string (str): text to modify

    Returns:
        str: modified text
    """
    result = ""
    for char in input_string:
        if not char.isalpha():
            break
        result += char
    return result


def tag_filter(tag: str) -> str:
    """
    return only the alphabetic part of the tag, if it is not alphabetic,
    return the tag until the first non-alphabetic character

    Args:
        tag (str): teg to be filtered

    Returns:
        str: the filtered tag
    """
    if tag.isalpha() or contains_only_nonalphabetic(tag):
        return tag
    else:
        return cut_until_nonalphabetic(tag)


# This function imports the database, adds the start token and make it list of words (and not sentences)
def import_db() -> (
    List[List[Tuple[str, str]], List[Tuple[str, str]], List[List[Tuple[str, str]]]]
):
    """create the training and test sets from the brown database

    Returns:
        List: [
            training_set_with_good_tags_flat:List[Tuple[word:str,tag:str]],
            test_set_with_good_tags_flat:[List[Tuple[word:str,tag:str]]],
            test_set_with_good_tags_not_flat:[List[sentence:List[Tuple[word:str,tag:str]]]]
            ]
    """

    brown_db = brown.tagged_sents(categories="news")
    # add start token for beginning of sentence
    brown_db_with_start = [start_token_tagged + sentence for sentence in brown_db]

    # make training set
    training_set = brown_db_with_start[: round(0.9 * len(brown_db_with_start))]
    training_set_flat = [item for sublist in training_set for item in sublist]
    training_set_with_good_tags_flat = [
        (word[0], tag_filter(word[1])) for word in training_set_flat
    ]

    # make test set
    test_set = brown_db_with_start[round(0.9 * len(brown_db_with_start)) :]
    test_set_flat = [item for sublist in test_set for item in sublist]
    test_set_with_good_tags_flat = [
        (word[0], tag_filter(word[1])) for word in test_set_flat
    ]
    test_set_with_good_tags_not_flat = [
        [(word[0], tag_filter(word[1])) for word in sublist] for sublist in test_set
    ]

    return [
        training_set_with_good_tags_flat,
        test_set_with_good_tags_flat,
        test_set_with_good_tags_not_flat,
    ]


def most_likely_baseline(training_set: List[str], tagged_word):

    training_set_quantities = Counter(training_set)
    max_of_occur = 0
    tag_prediction = UNKNOWN_WORD_TAG

    for word in training_set:
        if (
            word[WORD_POSITION] == tagged_word[WORD_POSITION]
            and training_set_quantities[word] > max_of_occur
        ):
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

    # making emission prob for unknown word
    tags_unknown_word = tags_possible
    tags_unknown_word[UNKNOWN_WORD_TAG] = 1
    hmm[UNKNOWN_WORD] = tags_unknown_word

    return hmm


def viterbi(sentence, tags, trans_prob, emit_prob, start_prob):
    """
    Viterbi algorithm for Hidden Markov Model (HMM).

    Parameters:
    - sentence: List of observed symbols
    - tags: List of hidden states
    - start_prob: Dictionary of initial state probabilities
    - trans_prob: 2D dictionary of transition probabilities between states
    - emit_prob: 2D dictionary of emission probabilities for each state and symbol

    Returns:
    - path: Most likely sequence of hidden states
    - max_prob: Probability of the most likely path
    """

    # Initialize the Viterbi matrix and backpointer matrix
    viterbi_mat = [{}]
    backpointer = [{}]
    # Initialization step
    for tag in tags:
        viterbi_mat[0][tag] = start_prob[tag] * emit_prob[tag][sentence[0]]
        backpointer[0][tag] = None

    # Recursion step
    for t in range(1, len(sentence)):
        viterbi_mat.append({})
        backpointer.append({})
        for state in tags:
            max_prob, prev_state = max(
                (
                    viterbi_mat[t - 1][prev_state]
                    * trans_prob[prev_state][state]
                    * emit_prob[state][sentence[t]],
                    prev_state,
                )
                for prev_state in tags
            )
            viterbi_mat[t][state] = max_prob
            backpointer[t][state] = prev_state

    # Termination step
    max_prob, state = max(
        (viterbi_mat[len(sentence) - 1][final_state], final_state)
        for final_state in tags
    )

    # Backtrack to find the most likely path
    path = [state]
    for t in range(len(sentence) - 1, 0, -1):
        state = backpointer[t][state]
        path.insert(0, state)

    return path, max_prob


class HMM:
    """Hidden Markov Model class,
    contains the transition and emission probabilities and the tags.
    filds:
    transitoin_prob:Dict[str,Dict[str,float]] - transition probabilities, for each tag a dictionary of the probabilities to the next tags.
    emission_prob:Dict[str,Dict[str,float]] - emission probabilities, for each word a dictionary of the probabilities to the tag.
    """

    def __init__(self, tages: List[str] = []):
        """
        Create a new HMM.

        Args:
            tages (List[str], optional): List of tags. Defaults to [].
        """
        self.transitoin_prob: Dict[str, Dict[str, float]] = {}
        self.emission_prob: Dict[str, Dict[str, float]] = {}
        self.transition_trained: bool = False
        self.emission_trained: bool = False
        self.tags: List[str] = tages

    def train_transition(self, training_set: List[Tuple[str, str]]) -> None:
        """treain the transition probabilities, for each tar a dictionary of the probabilities to the next tags.

        Args:
            training_set (List[Tuple[str, str]]): the training set, list of tuples of words and tags,
            first element is the word, second is the tag.
        """
        self.transitoin_prob = {
            tag1: {tag2: 0 for tag2 in self.tags} for tag1 in self.tags
        }
        for i in range(
            len(training_set) - 1
        ):  # counting the number of appearances of each bigram
            self.transitoin_prob[training_set[i][TAG_POSITION]][
                training_set[i + 1][TAG_POSITION]
            ] += 1
        for tag1 in self.tags:  # normalize
            sum_sub_dict = sum(self.transitoin_prob[tag1].values())
            for tag2 in self.transitoin_prob[tag1]:
                self.transitoin_prob[tag1][tag2] /= sum_sub_dict
        self.transition_trained = True

    def train_emission(self, training_set: List[Tuple[str, str]]) -> None:
        """treain the emission probabilities,
        for each word a dictionary of the probabilities to the tag.

        Args:
            training_set (List[Tuple[str, str]]): the training set,
            list of tuples of words and tags, first element is the word, second is the tag.
        """
        self.emission_prob = {
            word: {tag: 0 for tag in self.tags} for word in training_set
        }
        for word in training_set:  # counting the number of appearances of each word-tag
            self.emission_prob[word[WORD_POSITION]][word[TAG_POSITION]] += 1
        for word_emission in self.emission_prob.values():  # normalize
            for tag in word_emission:
                word_emission[tag] /= sum(word_emission.values())
        self.emission_trained = True

    def __get_best_tag(self, word: str, last_tag: str) -> (str, float):
        """get the most likely tag for the word given the last tag.

        Args:
            word (str): the word.
            last_tag (str): the last tag.

        Returns:
            (str,float): the most likely tag for the word given the last tag, and the probabilti.
        """
        best_tag = max(
            self.tags,
            key=lambda tag: self.transitoin_prob[last_tag][tag]
            * self.emission_prob[word][tag],
        )
        return (
            best_tag,
            self.transitoin_prob[last_tag][best_tag]
            * self.emission_prob[word][best_tag],
        )

    def viterbi(self, sentence: List[str]) -> (List[str], float):
        """aplly the viterbi algorithm on sentence.
        find the most likely sequence of tags for the sentence.

        Algorithm:
        backpointer - List[List[List,float]] in every iteration contian the most likely sequence of tags until the current word,
        and the probability of this sequence.
        a. Init backpointer.
        b. for each word:
            b.1. for each last tag:
                b.1.1. find the next tag with max probability given by the transition and emission probabilities.
                b.1.2. update the backpointer with the max probability and the next tag, and the probabilits.
        c. find the most likely sequence of tags.

        Args:
            sentence (List[str]): list of words, represent the sentence.
        Returns:
            (List[str], float): the most likely sequence of tags for the sentence, and the probability of this sequence.
        """
        if not self.transition_trained or not self.emission_trained:
            raise ValueError(
                "The transition and emission probabilities must be trained"
            )
        backpointer = {tag: [[], 0] for tat in self.tags}
        for tag in self.tags:  # init the first word in the sentence
            backpointer[tag][0].append(tag)
            backpointer[tag][1] = self.emission_prob[sentence[0]][tag]
        for i in range(1, len(sentence)):
            for tag in self.tags:
                best_tag, best_prob = self.__get_best_tag(sentence[i], tag)
                backpointer[tag][0].append(best_tag)
                backpointer[tag][1] = best_prob
        best_last_tag = max(backpointer, key=lambda tag: backpointer[tag][1])
        return backpointer[best_last_tag][0], backpointer[best_last_tag][1]


def main():
    [training_set_flat, test_set_flat, test_set_senteced] = import_db()

    if SECTION == "b":
        # b section - DONE
        test_set_unique_words = list(set(tuple(t) for t in test_set_flat))
        test_set_unique_words.remove(start_token_tagged[0])
        num_of_words = len(test_set_unique_words)
        i = 0
        correct_tags_known_words = 0
        amount_known_words = 0
        correct_tags_unknown_words = 0
        amount_unknown_words = 0
        for word in test_set_unique_words:
            i += 1
            if word_in_training_set_checker(training_set_flat, word):
                if word[TAG_POSITION] == most_likely_baseline(training_set_flat, word):
                    correct_tags_known_words += 1
                amount_known_words += 1
            else:
                if word[TAG_POSITION] == most_likely_baseline(training_set_flat, word):
                    correct_tags_unknown_words += 1
                amount_unknown_words += 1

        print(
            "The error rate for known words is   ",
            1 - correct_tags_known_words / amount_known_words,
        )
        print(
            "The error rate for unknown words is ",
            1 - correct_tags_unknown_words / amount_unknown_words,
        )
        print(
            "The total error rate is ",
            1
            - (correct_tags_known_words + correct_tags_unknown_words)
            / (amount_unknown_words + amount_known_words),
        )

    # c section
    elif SECTION == "c":
        # transition_hmm = train_transition_hmm(training_set_flat)
        # emission_hmm = train_emission_hmm(training_set_flat)
        # tags = list({key[TAG_POSITION] for key in training_set_flat})
        # start_prob = {key[TAG_POSITION]: 1 for key in training_set_flat}

        # [path, max_prob] = viterbi(
        #     test_set_senteced[0], tags, transition_hmm, emission_hmm, start_prob
        # )

        # print["path: ", path]
        


if __name__ == "__main__":
    main()
