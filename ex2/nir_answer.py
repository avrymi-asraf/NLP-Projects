from collections import defaultdict, Counter

# Constants
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNKNOWN_WORD_TAG = "NN"  # setting the tag of unknown like from the question


class BigramHMMTagger:
    def __init__(self, is_delta=False, is_pseudo=False):
        self.is_delta = is_delta
        self.is_pseudo = is_pseudo
        self.transitions_probs = defaultdict(lambda: defaultdict(float))
        self.emissions_probs = defaultdict(lambda: defaultdict(float))
        self.prevs_tags_counter = Counter()

    def compute_transition_probs(self, tagged_sentences):
        """
        This function computes the transition probabilities of the HMM model.
        :param tagged_sentences: a list of sentences, where each sentence is a list of tuples (word, tag)
        :return: a dictionary where the keys are the previous tags and the values are dictionaries of the next tags
        and their probabilities.
        """
        tags_counter = defaultdict(Counter)
        for sentence in tagged_sentences:
            prev_tag = START_TAG
            for word, tag in sentence:
                tags_counter[prev_tag][tag] += 1
                self.prevs_tags_counter[prev_tag] += 1
                prev_tag = tag
                # handle the STOP_TAG
            tags_counter[prev_tag][STOP_TAG] += 1
            self.prevs_tags_counter[prev_tag] += 1
        for prev_tag, tags in tags_counter.items():
            for tag, count in tags.items():
                self.transitions_probs[prev_tag][tag] = (
                    count / self.prevs_tags_counter[prev_tag]
                )

    def compute_emission_probs(self, tagged_sentences, all_words, delta):
        """
        This function computes the emission probabilities of the HMM model.
        :param tagged_sentences: a list of sentences, where each sentence is a list of tuples (word, tag)
        :param all_words: a list of all words in the train and test set
        :param delta: a smoothing parameter (default value is 0)
        :return: a dictionary where the keys are the tags and the values are dictionaries of the words and their
        probabilities.
        """
        words_tags_counter = defaultdict(Counter)
        for sentence in tagged_sentences:
            for word, tag in sentence:
                words_tags_counter[tag][word] += 1
        for word in all_words:
            for tag in words_tags_counter:
                self.emissions_probs[tag][word] = (
                    words_tags_counter[tag][word] + delta
                ) / (self.prevs_tags_counter[tag] + delta * len(all_words))

    def train(self, tagged_sentences, all_words=None, delta=0):
        """
        Train the HMM model using the given tagged sentences
        :param tagged_sentences: a list of sentences, where each sentence is a list of tuples (word, tag)
        :param all_words: a list of all words in the train and test set
        :param delta: a smoothing parameter (default value is 0)
        :return: None
        """
        self.compute_transition_probs(tagged_sentences)
        self.compute_emission_probs(tagged_sentences, all_words, delta)

    def viterbi_algorithm(self, sentence, train_word_count_dict):
        """
        This function computes the most likely sequence of tags for the given sentence using the Viterbi algorithm.
        :param sentence: a list of words - the sentence to predict tags for
        :param train_word_count_dict: a dictionary of words and their counts in the training set
        :return: a list of tags - predicted tags for the sentence
        """
        S = list(self.emissions_probs.keys())
        stored_probs = []
        # run for each word in the sentence
        for k in range(len(sentence)):
            current_word = sentence[k][0]  # tuple (word, tag)
            stored_probs.append([])
            if (
                current_word not in train_word_count_dict
                and (not self.is_pseudo)
                and (not self.is_delta)
            ):
                self.emissions_probs[UNKNOWN_WORD_TAG][current_word] = 1
            for i, v in enumerate(S):
                # in case of the first word in the sentence
                if k == 0:
                    stored_probs[k].append(
                        (
                            self.transitions_probs[START_TAG][v]
                            * self.emissions_probs[v][current_word],
                            i,
                        )
                    )
                else:
                    all_probs = []
                    for j, u in enumerate(S):
                        optional_prob = (
                            stored_probs[k - 1][j][0]
                            * self.transitions_probs[u][v]
                            * self.emissions_probs[v][current_word]
                        )
                        all_probs.append(optional_prob)
                    max_prob = max(all_probs)
                    max_prob_tag_index = all_probs.index(max_prob)
                    stored_probs[k].append((max_prob, max_prob_tag_index))
        optimal_tags = []
        for index, last_tag in enumerate(S):
            optimal_tags.append(
                stored_probs[-1][index][0] * self.transitions_probs[last_tag][STOP_TAG]
            )
        prev_tag_index = optimal_tags.index(max(optimal_tags))
        predicted_tags = [S[prev_tag_index]]
        for k in range(len(sentence) - 1, 0, -1):
            tag = S[stored_probs[k][prev_tag_index][1]]
            predicted_tags.insert(0, tag)
            prev_tag_index = stored_probs[k][prev_tag_index][1]
        return predicted_tags

    def compute_viterbi_error_rates(
        self, test_set, train_word_count_dict, pseudo_test_set=None
    ):
        """
        This function computes the error rates of the Viterbi algorithm on the given test set.
        :param test_set: The test set - a list of sentences, where each sentence is a list of tuples (word, tag)
        :param train_word_count_dict: a dictionary of words and their counts in the training set
        :param pseudo_test_set: pseudo words test set.
        :return: a tuple of the total error rate, the known word error rate, and the unknown word error rate
        """
        known_word_errors = 0
        unknown_word_errors = 0
        known_word_counter = 0
        unknown_word_counter = 0
        for j, sentence in enumerate(test_set):
            if self.is_pseudo:
                predicted_tags = self.viterbi_algorithm(
                    pseudo_test_set[j], train_word_count_dict
                )
            else:
                predicted_tags = self.viterbi_algorithm(sentence, train_word_count_dict)
        for i, (word, correct_tag) in enumerate(sentence):
            # in case the word is in the most_likely_tags dictionary
            if word in train_word_count_dict:
                known_word_counter += 1
            if correct_tag != predicted_tags[i]:
                known_word_errors += 1
            else:
                unknown_word_counter += 1
            if correct_tag != predicted_tags[i]:
                unknown_word_errors += 1
        known_error_rate = (
            known_word_errors / known_word_counter if known_word_counter > 0 else 0
        )
        unknown_error_rate = (
            unknown_word_errors / unknown_word_counter
            if unknown_word_counter > 0
            else 0
        )
        total_error_rate = (
            (known_word_errors + unknown_word_errors)
            / (known_word_counter + unknown_word_counter)
            if (known_word_counter + unknown_word_counter) > 0
            else 0
        )
        return total_error_rate, known_error_rate, unknown_error_rate


