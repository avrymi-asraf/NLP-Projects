import numpy as np
import nltk as nl
from nltk.corpus import brown
from collections import Counter
from typing import List, Optional, Set, Tuple, Dict, Union, Callable
import re
import builtins
from math import exp, log


def custom_print(*args, **kwargs):
    new_args = [(round(arg, 3) if isinstance(arg, float) else arg) for arg in args]
    builtins.print(*new_args, **kwargs, flush=True)


point = lambda: print(".", end="", flush=True)


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


def get_pos_tag(tag):
    """
    Extracts the POS tag from a complex tag using regular expressions, handling special characters.

    Args:
      tag: The complex tag string.

    Returns:
      The POS tag, or None if no POS tag is found.
    """
    match = re.search(r"([^+\-]+)", tag)
    if match:
        return match.group(1)
    else:
        return ""


def new_import_db(flat=False, ratio=0.9, category="news"):
    """if flat is True return the training and test sets as flat lists, otherwise as list of sentences.
    and add start token for beginning of sentence
    """
    brown_db = brown.tagged_sents(categories=category)
    # add start token for beginning of sentence
    brown_db_with_start = [start_token_tagged + sentence for sentence in brown_db]
    # make training set
    training_set = brown_db_with_start[: round(0.9 * len(brown_db_with_start))]
    test_set = brown_db_with_start[round(0.9 * len(brown_db_with_start)) :]
    training_set = [
        [(word[0], get_pos_tag(word[1])) for word in sentence]
        for sentence in training_set
    ]
    test_set = [
        [(word[0], get_pos_tag(word[1])) for word in sentence] for sentence in test_set
    ]
    if flat:
        training_set = [item for sublist in training_set for item in sublist]
        test_set = [item for sublist in test_set for item in sublist]
    return training_set, test_set

numbers_patters = {
    "<MONEY>": re.compile(r"(\$[\d,.]+)|([\d,.]+\$)"),
    "<TowDigits>": re.compile(r"\d{2}"),
    "<ThreeDigits>": re.compile(r"\d{3}"),
    "<FourDigits>": re.compile(r"\d{4}"),
    "<LongDigits>": re.compile(r"\d{5,}"),
    "<DigitsAndAlpha>": re.compile(r"[A-Za-z]\d+"),
    "<DigitAndDash>": re.compile(r"[\d,.]+(-[\d,.])+"),
    "<DigitsAndPeriod>": re.compile(r"^\d+\.\d+"),
    "<AllCaps>": re.compile(r"[A-Z]{2,}"),
    "<OrdNum>": re.compile(r"\d[\d,.]*(nd|th|st)"),
    "<Years>": re.compile(r"\d+(\'s)"),
    "<Percentage>": re.compile(r"\d[\d,.]*%"),
    "<Units>": re.compile(r"\d[\d,.]*-[A-Za-z]+"),
    "<WordsAndDash>": re.compile(r"[A-Za-z]+-([A-Za-z]+)"),
}


def filter_psudo_words(word: str):
    for psudo_word, pattern in numbers_patters.items():
        if pattern.search(word):
            return psudo_word
    return word


class HMM:
    """Hidden Markov Model class,
    contains the transition and emission probabilities and the tags.
    filds:
    transitoin_prob:Dict[str,Dict[str,float]] - transition probabilities, for each tag a dictionary of the probabilities to the ext tags.
    emission_prob:Dict[str,Dict[str,float]] - emission probabilities, for each word a dictionary of the probabilities to the tag.
    """

    def __init__(
        self,
        tages: List[str] = [],
        filter_words: Optional[Callable[[str], str]] = None,
    ):
        """
        Create a new HMM
        Args:
            tages (List[str], optional): List of tags. Defaults to [].
            replace_words (Optional[Callable[[str], str]]): if functoin is not None, every word that pass by this word.
        """
        self.transitoin_prob_: Dict[str, Dict[str, float]] = {}
        self.emission_prob_dict: Dict[str, Dict[str, float]] = {}
        self.transition_trained: bool = False
        self.emission_trained: bool = False
        self.tags: set[str] = set(tages)
        self.filter_words = filter_words

    def __contains__(self, word: str) -> bool:
        """return if word is in model

        Args:
            word (str): word to check

        Returns:
            bool: if word is in model or not
        """
        return bool(self.emission_prob_dict.get(word, False))

    def train_transition(
        self, training_set: List[Tuple[str, str]], log_space: bool = True
    ):
        """treain the transition probabilities,
        for each tag a dictionary of the probabilities to the next tags.
        Args:
            training_set (List[Tuple[str, str]]): the training set, list of tuples of words and tags,
            first element is the word, second is the tag.
            log_space (bool): if calculte in log space.
        """
        self.transitoin_prob_ = {
            tag1: {tag2: 0.0 for tag2 in self.tags} for tag1 in self.tags
        }
        for i in range(len(training_set) - 1):
            # if training_set[i + 1][WORD_POSITION] == START_TOKEN:
            #     continue
            self.transitoin_prob_[training_set[i][TAG_POSITION]][
                training_set[i + 1][TAG_POSITION]
            ] += 1
        if log_space:
            for tag1 in self.tags:  # normalize
                sum_sub_dict = sum(self.transitoin_prob_[tag1].values())
                for tag2 in self.transitoin_prob_[tag1]:
                    if self.transitoin_prob_[tag1][tag2]:
                        self.transitoin_prob_[tag1][tag2] = log(
                            self.transitoin_prob_[tag1][tag2] / sum_sub_dict
                        )
                    else:
                        self.transitoin_prob_[tag1][tag2] = float("-inf")
        else:
            for tag1 in self.tags:  # normalize
                sum_sub_dict = sum(self.transitoin_prob_[tag1].values())
                for tag2 in self.transitoin_prob_[tag1]:
                    self.transitoin_prob_[tag1][tag2] = (
                        self.transitoin_prob_[tag1][tag2] / sum_sub_dict
                    )

        self.transition_trained = True
        return self

    def train_emission(
        self, training_set: List[Tuple[str, str]], delta=0, log_space=True
    ) -> None:
        """treain the emission probabilities,
        for each word a dictionary of the probabilities to the tag.
        Args:
            training_set (List[Tuple[str, str]]): the training set,
            list of tuples of words and tags, first element is the word, second is the tag.
        """
        # init start and unknown word
        if self.filter_words:
            training_set = [
                (self.filter_words(word), tag) for (word, tag) in training_set
            ]
        uniqe_words = {word for (word, tag) in training_set}.union(
            {START_TOKEN, UNKNOWN_WORD}
        )

        self.emission_prob_dict = {
            word: {tag: 0 for tag in self.tags} for word in uniqe_words
        }
        counter = {tag: 0 for tag in self.tags}
        for word in training_set:  # counting the number of appearances of each word-tag
            self.emission_prob_dict[word[WORD_POSITION]][word[TAG_POSITION]] += 1
            counter[word[TAG_POSITION]] += 1
        if log_space:
            for word_emission in self.emission_prob_dict.values():  # normalize
                for tag in word_emission:
                    if word_emission[tag]:
                        word_emission[tag] = log(
                            (word_emission[tag] + delta)
                            / (counter[tag] + delta * len(uniqe_words))
                        )
                    else:
                        word_emission[tag] = float("-inf")
            self.emission_prob_dict[START_TOKEN][START_TAG] = log(1)
            self.emission_prob_dict[UNKNOWN_WORD][UNKNOWN_WORD_TAG] = log(1)
        else:
            for word_emission in self.emission_prob_dict.values():  # normalize
                for tag in word_emission:
                    word_emission[tag] = word_emission[tag] + delta / (
                        counter[tag] + delta * len(uniqe_words)
                    )

        # TODO: what we do with the start token?
        self.emission_trained = True

    def predict_tag(self, word: str) -> str:
        """predict the most likely tag for given word

        Args:
            word (str): word to get the prdiction

        Returns:
            str: the tag most likelky be, if word not in db return "UNKNOW_TAG"
        """
        if not self.emission_trained:
            raise ValueError(
                "The transition and emission probabilities must be trained"
            )
        word = self.filter_words(word) if self.filter_words else word
        if word not in self:
            return "NN"
        return max(self.emission_prob_dict[word], key=self.emission_prob_dict[word].get)

    def emission_prob(self, word: str, tag: str) -> float:
        """get the emission probability of the word given the tag.
        Args:
            word (str): the word.
            tag (str): the tag.
        Returns:
            float: the emission probability of the word given the tag.
        """
        if not self.emission_trained:
            raise ValueError("The emission probabilities must be trained")
        word = self.filter_words(word) if self.filter_words else word
        if word in self:
            return self.emission_prob_dict[word][tag]
        else:
            if tag == UNKNOWN_WORD_TAG:
                return 0.0
            return float("-inf")

    def __get_best_tag(
        self,
        word: str,
        tag: str,
        backpointer: Dict[str, List[Union[list, float]]],
        log_space=True,
    ) -> Tuple[List[str], float]:
        """get the most likely sequnce of tags for given tag and word.
        Args:
            tag (str): The tag for him needs to find the best sequnce.
            word (str): The given word.
            backpointer (Dict[str:List[List,float]]): the backpointer,
            for each tag contain the most likely sequence of tags ended with this tag,
            and the probability of this sequence.
        Returns:
            (List[str],float): the most likely sqeunce tags ending by tag, and the probabilti.
        """
        if log_space:
            best_tag = max(
                backpointer,
                key=lambda t: self.transitoin_prob_[t][tag] + backpointer[t][1],
            )
            return backpointer[best_tag][0] + [tag], self.transitoin_prob_[best_tag][
                tag
            ] + backpointer[best_tag][1] + self.emission_prob(word, tag)
        else:
            best_tag = max(
                backpointer,
                key=lambda t: self.transitoin_prob_[t][tag] * backpointer[t][1],
            )
            return backpointer[best_tag][0] + [tag], self.transitoin_prob_[best_tag][
                tag
            ] * backpointer[best_tag][1] * self.emission_prob(word, tag)

    def viterbi(self, sentence: List[str], log_space=True) -> Tuple[List[str], float]:
        """
        aplly the viterbi algorithm on sentence.
        find the most likely sequence of tags for the sentence.
        Algorithm:
            backpointer - Dict[str:List[List,float]]
            in every iteration contian the most likely sequence of tags until the current word,
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
            (List[str], float): the most likely sequence of tags for the sentence,
             and the probability of this sequence.
        """
        if not self.transition_trained or not self.emission_trained:
            raise ValueError(
                "The transition and emission probabilities must be trained"
            )
        if self.filter_words:
            sentence = [self.filter_words(word) for word in sentence]
        backpointer = {tag: [[], 0] for tag in self.tags}
        for tag in self.tags:
            backpointer[tag][0].append(tag)
            backpointer[tag][1] = self.emission_prob(sentence[0], tag)
        for i in range(1, len(sentence)):
            next_level = {}
            for tag in self.tags:
                best_seq, best_prob = self.__get_best_tag(
                    sentence[i], tag, backpointer, log_space=log_space
                )
                next_level[tag] = [best_seq, best_prob]
            for tag in self.tags:
                backpointer[tag] = next_level[tag]
        best_last_tag = max(backpointer, key=lambda tag: backpointer[tag][1])
        return backpointer[best_last_tag][0], backpointer[best_last_tag][1]

    def error_rate(
        self, sentence: List[str], target: List[str]
    ) -> Tuple[float, float, float]:
        """calculate the error rate for the sentence.
        return the error rate for known words, unknown words, and the total error rate.

        Args:
            sentence (List[str]): the sentence.
            target (List[str]): the target tags of the sentence.

        Returns:
            Tuple[float,float,float]: error rate for known words, unknown words, and the total error rate.
        """
        pred = self.viterbi(sentence)
        known_words = 0
        unknown_words = 0
        correct_known_words = 0
        correct_unknown_words = 0
        for word, tag, pred_tag in zip(sentence, target, pred):
            if word in self:
                known_words += 1
                if tag == pred_tag:
                    correct_known_words += 1
            else:
                unknown_words += 1
                if tag == pred_tag:
                    correct_unknown_words += 1
        return (
            1 - correct_known_words / known_words,
            1 - correct_unknown_words / unknown_words,
            1
            - (correct_known_words + correct_unknown_words)
            / (known_words + unknown_words),
        )

    def error_rate_corpus(
        self, sentences: List[List[Tuple[str, str]]]
    ) -> Tuple[float, float, float]:
        """return the error rate for the corpus. gives as list of sentences. all sentences are list of tuples of words and tags.

        Args:
            sentences (List[List[Tuple[str, str]]]): corpus text

        Returns:
            Tuple[float, float, float]: error rate for known words, unknown words, and the total error rate.
        """
        known_words = 0
        unknown_words = 0
        correct_known_words = 0
        correct_unknown_words = 0
        for sentence in sentences:
            pred, prob = self.viterbi([word for word, tag in sentence])
            for (word, tag), pred_tag in zip(sentence, pred):
                if word in self:
                    known_words += 1
                    if tag == pred_tag:
                        correct_known_words += 1
                else:
                    unknown_words += 1
                    if tag == pred_tag:
                        correct_unknown_words += 1
        return (
            1 - correct_known_words / known_words,
            1 - correct_unknown_words / unknown_words,
            1
            - (correct_known_words + correct_unknown_words)
            / (known_words + unknown_words),
        )


def get_all_tags(training_set: List[Tuple[str, str]]) -> Set[str]:
    """
    Return list with all tags in the training set

    Args:
        raining_set (List[Tuple[str, str]]): training set

    Returns:
        Set[str]: Set with all tags in the training set
    """
    return {key[TAG_POSITION] for key in training_set}.union(
        {
            START_TAG,
            UNKNOWN_WORD_TAG,
        }
    )


def main(section: str = "bcde"):
    print("import data base")
    training_set_sentence, test_set_senteced = new_import_db()
    training_set_flat, test_set_flat = new_import_db(flat=True)
    if "b" in section:
        # b section - DONE
        print("(b) Implementation of the most likely tag baseline:", flush=True, end="")
        tags = get_all_tags(training_set_flat)
        model = HMM(tags)
        model.train_emission(training_set_flat)
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
            if word[WORD_POSITION] in model:
                if word[TAG_POSITION] == model.predict_tag(word[WORD_POSITION]):
                    correct_tags_known_words += 1
                amount_known_words += 1
            else:
                if word[TAG_POSITION] == model.predict_tag(word[WORD_POSITION]):
                    correct_tags_unknown_words += 1
                amount_unknown_words += 1

        total_error_rate_b_part = 1 - (
            correct_tags_known_words + correct_tags_unknown_words
        ) / (amount_unknown_words + amount_known_words)
        custom_print(
            "\t The error rate for known words is   ",
            1 - correct_tags_known_words / amount_known_words,
        )
        custom_print(
            "\t The error rate for unknown words is ",
            1 - correct_tags_unknown_words / amount_unknown_words,
        )
        custom_print(
            "\t The total error rate is ",
            1
            - (correct_tags_known_words + correct_tags_unknown_words)
            / (amount_unknown_words + amount_known_words),
        )

    if "c" in section:
        print("\n(c) Implementation of a bigram HMM tagger:", flush=True)
        tags = get_all_tags(training_set_flat)
        hmm = HMM(tags)
        hmm.train_emission(training_set_flat)
        hmm.train_transition(training_set_flat)
        known_error_rate, unknown_error_rate, total_error_rate = hmm.error_rate_corpus(
            test_set_senteced
        )
        custom_print("\t The error rate for known words is   ", known_error_rate)
        custom_print("\t The error rate for unknown words is ", unknown_error_rate)
        custom_print("\t The total error rate is             ", total_error_rate)
    
    if "d" in section:
        print("\n(d) Using Add-one smoothing", flush=True)
        tags = get_all_tags(training_set_flat)
        hmm = HMM(tags)
        hmm.train_emission(training_set_flat, delta=1)
        hmm.train_transition(training_set_flat)
        known_error_rate, unknown_error_rate, total_error_rate = hmm.error_rate_corpus(
            test_set_senteced
        )

        custom_print("\t The error rate for known words is   ", known_error_rate)
        custom_print("\t The error rate for unknown words is ", unknown_error_rate)
        custom_print("\t The total error rate is             ", total_error_rate)

    if "e" in section:
        print("\n(e) Using pseudo-words", flush=True)
        tags = get_all_tags(training_set_flat)
        hmm = HMM(tags, filter_psudo_words)
        hmm.train_emission(training_set_flat)
        hmm.train_transition(training_set_flat)
        known_error_rate, unknown_error_rate, total_error_rate = hmm.error_rate_corpus(
            test_set_senteced
        )
        custom_print("\t The error rate for known words is   ", known_error_rate)
        custom_print("\t The error rate for unknown words is ", unknown_error_rate)
        custom_print("\t The total error rate is             ", total_error_rate)


        print("\n(d) Using Add-one smoothing and pseudo-words", flush=True)
        tags = get_all_tags(training_set_flat)
        hmm = HMM(tags, filter_psudo_words)
        hmm.train_emission(training_set_flat,delta=1)
        hmm.train_transition(training_set_flat)
        known_error_rate, unknown_error_rate, total_error_rate = hmm.error_rate_corpus(
            test_set_senteced
        )
        custom_print("\t The error rate for known words is   ", known_error_rate)
        custom_print("\t The error rate for unknown words is ", unknown_error_rate)
        custom_print("\t The total error rate is             ", total_error_rate)


if __name__ == "__main__":
    main()
