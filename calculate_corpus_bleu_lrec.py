"""


"""
import git
import os
from nltk.translate.bleu_score import corpus_bleu

from spacy.lang.en import English

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer

repo = git.Repo('.', search_parent_directories=True)
root_repo_dir = repo.working_tree_dir


def references_read_in(path_to_file_references):
    """ The references (human-written summaries) take the following format: one summary per line; the
     summaries belonging to the same chart are followed by each other. Blank lines separate between charts. Example:

         chart1_summary1
         chart1_summary2

         chart2_summary1

         chart3_summary1
    """
    list_of_references = []
    with open(path_to_file_references) as f:
        current_chart = []
        for line in f:
            line = line.strip()  # still a string
            if not line:  # line was \n
                list_of_references.append(current_chart)  # append to the collected references and redefine current
                current_chart = []
                continue
            if line:
                current_tokens = [t for t in tokenizer(line) if t]
                current_chart.append(current_tokens)

    return list_of_references


def candidates_read_in(path_to_file_candidates):
    """ The outputs are one summary per line """
    list_of_candidates = []
    with open(path_to_file_candidates) as f:
        for line in f:
            line = line.strip()
            tokens = [t for t in tokenizer(line) if t]
            list_of_candidates.append(tokens)

    return list_of_candidates


def calculate_corpus_bleu(list_references, list_candidates):
    """ The input to the corpus_bleu function must be tokenized """
    assert len(list_references) == len(list_candidates)
    score = corpus_bleu(list_references, list_candidates)
    return score * 100

# types: mixed-basic mixed-inferred
#       exclusive-basic exclusive-inferred


path_references = ""
path_candidates = ""

if __name__ == "__main__":
    bleu = calculate_corpus_bleu(references_read_in(path_references), candidates_read_in(path_candidates))
    print("BLEU-4 score %f for %s and %s" % (bleu, path_candidates, path_references))

