import json
import spacy
from functools import reduce
from utils import load_dataset, save_dataset, print_progress

nlp = spacy.load("en_core_web_sm")

def replace_nouns(phrase, other_object="NaN"):
    doc = nlp(phrase)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    new_phrase = phrase
    for noun in nouns:
        new_phrase = new_phrase.replace(noun, other_object)
    return new_phrase

def replace_chunk_noun(phrase, other_object="NaN"):
    doc = nlp(phrase)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    new_phrase = phrase
    for chunk in noun_chunks:
        new_phrase = new_phrase.replace(chunk, other_object)
    return new_phrase

def change_nouns(split):
    data = load_dataset(split)
    changed_data = []
    total_length = len(data)

    print(f"Changing nouns words for {split}")
    for i, element in enumerate(data):
        instructions = element["instructions"]
        changed_instructions = [replace_nouns(instr) for instr in instructions]
        element["instructions"] = changed_instructions
        changed_data.append(element)
        print_progress(i + 1, total_length, prefix='Progress:', suffix='Complete', bar_length=50)

    save_dataset(split, changed_data, "nouns")

def change_chunk_nouns(split):
    data = load_dataset(split)
    changed_data = []
    total_length = len(data)

    print(f"Changing chunk nouns for {split}")
    for i, element in enumerate(data):
        instructions = element["instructions"]
        changed_instructions = [replace_chunk_noun(instr) for instr in instructions]
        element["instructions"] = changed_instructions
        changed_data.append(element)
        print_progress(i + 1, total_length, prefix='Progress:', suffix='Complete', bar_length=50)

    save_dataset(split, changed_data, "chunks")

def change_all_splits():
    for split in ['train', 'val_seen', 'val_unseen', 'test', 'synthetic']:
        change_nouns(split)
        change_chunk_nouns(split)

if __name__ == "__main__":
    change_all_splits()