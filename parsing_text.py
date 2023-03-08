# Parsing is an NLP process concerned with segmenting text based on syntax.
# part of speech tagging, Named entity recognition

import spacy
from nltk import Tree
from squids import squids_text

dependency_parser = spacy.load('en')
parsed_squids = dependency_parser(squids_text)


my_sentence = "Your sentence goes here!"
my_parsed_sentence = dependency_parser(my_sentence)


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
        return Tree(node.orth_, parsed_child_nodes)
    else:
        return node.orth_    


for sent in parsed_squids.snets:
    to_nltk_tree(sent.root).pretty_print()

for sent in my_parsed_sentence.sents:
    to_nltk_tree(sent.root).pretty_print()