import unittest
import sys
sys.path.append("src/")

from nlp.spacy_helper import *
from helpers.print_helper import *

class SpacyHelperTest(unittest.TestCase):
    def setUp(self):
        self.vocab = ["word1", "word2", " this is a !@#$ with word2"]

    def test_naive_vocab_creater(self):
        if os.path.exists("/tmp/vocab.txt"):
            print_info("Deleting /tmp/vocab.txt")
            os.remove("/tmp/vocab.txt")
        self.pre_num_tokens, self.prev_final_vocab = naive_vocab_creater(out_file_name="/tmp/vocab.txt", lines=self.vocab, use_nlp=True)
        self.assertEqual(self.pre_num_tokens, 11)
        self.assertEqual(self.prev_final_vocab,  ['<PAD>', '<UNK>', ' ', '!', '@#$', 'a', 'is', 'this', 'with', 'word1', 'word2'])

    def test_naive_vocab_creater_read_back(self):
        self.current_num_tokens, self.current_final_vocab = naive_vocab_creater(out_file_name="/tmp/vocab.txt", lines=None, use_nlp=None)
        self.assertEqual(11, self.current_num_tokens)
        self.assertEqual(['<PAD>', '<UNK>', ' ', '!', '@#$', 'a', 'is', 'this', 'with', 'word1', 'word2'], self.current_final_vocab)

