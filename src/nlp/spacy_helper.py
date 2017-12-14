import spacy
from tqdm import tqdm
from helpers.print_helper import *
from tensorflow.python.platform import gfile
from config.global_constants import *


def naive_vocab_creater(lines, out_file_name, vocab_filter):
    nlp = spacy.load('en_core_web_md')
    final_vocab = [PAD_WORD, UNKNOWN_WORD]
    if vocab_filter:
        vocab = [word for line in tqdm(lines) for word in line.split(" ") if word in nlp.vocab]
    else:
        vocab = [word for line in tqdm(lines) for word in line.split(" ")]

    vocab = set(vocab)

    try:
        vocab.remove(UNKNOWN_WORD)
    except:
        print("No {} token found".format(UNKNOWN_WORD))

    vocab = list(vocab)
    final_vocab.extend(vocab)

    print_warn(out_file_name)

    # Create a file and store the words
    with gfile.Open(out_file_name, 'wb') as f:
        for word in final_vocab:
                f.write("{}\n".format(word))
    return len(final_vocab), final_vocab


def vocab_to_tsv(vocab_list, outfilename):
    '''

    :param vocab_list:
    :return:
    '''
    with gfile.Open(outfilename, 'wb') as file:
        for word in tqdm(vocab_list):
            if len(word) > 0:
                file.write("{}\n".format(word))

    nwords = len(vocab_list)
    print('{} words into {}'.format(nwords, outfilename))

    return nwords

def get_char_vocab(words_vocab):
    '''

    :param words_vocab: List of words
    :return:
    '''
    chars = set()
    for word in words_vocab:
        for char in word:
            chars.add(str(char))
    return sorted(chars)