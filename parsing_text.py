# Parsing is an NLP process concerned with segmenting text based on syntax.
# part of speech tagging, Named entity recognition

import spacy
from nltk import Tree
from squids import squids_text

dependency_parser = spacy.load('en_core_web_sm')
parsed_squids = dependency_parser(squids_text)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        parsed_child_nodes = [to_nltk_tree(child) for child in node.children]
        return Tree(node.orth_, parsed_child_nodes)
    else:
        return node.orth_    

def parser(sentence):
    parser = spacy.load('en_core_web_sm')
    parsed_sentence = parser(sentence)
    for sent in parsed_sentence.sents:
        to_nltk_tree(sent.root).pretty_print()


parser(parsed_squids)