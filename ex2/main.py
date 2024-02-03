#import torch
import numpy as np
import nltk as nl
from nltk.corpus import brown

POS_INDEX = 1
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



#nl.download('brown')
#brown = nl.corpus.brown
my_brown = brown.tagged_sents(categories='news')

#### Need to use only tags with tag_filter function!!!
training_set = my_brown[:round(0.9*len(my_brown))]
test_set = my_brown[round(0.9*len(my_brown)):]


pass