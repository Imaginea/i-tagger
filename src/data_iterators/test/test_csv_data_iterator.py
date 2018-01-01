import unittest
import sys
sys.path.append("")
sys.path.append("src/")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import numpy as np

from helpers.print_helper import *
from config.global_constants import *
from data_iterators.csv_data_iterator import CsvDataIterator

class CsvDataIteratorTest(unittest.TestCase):

    def setUp(self):
        print_info("==============")
        self.data_iterator = CsvDataIterator("conll_csv_experiments/", 32)

    def test_pad_sequences_level_one(self):
        seq = [["word1"], ["word1", "w2"], ["word1", "w2", "wd3"]]
        seq = [ "{}".format(SEPERATOR).join(words) for words in seq]

        sequence_padded, sequence_length = self.data_iterator._pad_sequences(sequences=seq,
                                                pad_tok="{}{}".format(SEPERATOR, PAD_WORD),
                                                nlevels=1)

        np.testing.assert_array_equal(sequence_length, np.array([3,3,3]))
        self.assertEqual(sequence_padded[0], 'word1'+SEPERATOR+PAD_WORD+SEPERATOR+PAD_WORD)
        self.assertEqual(sequence_padded[1], 'word1~w2'+SEPERATOR+PAD_WORD)
        self.assertEqual(sequence_padded[2], 'word1~w2~wd3')

    def test_pad_sequences_level_two(self):
        char_2_id_map = {"w": 0, "o": 1, "r": 2, "d": 3, "1": 4, "2": 5, "3": 6}
        sequences = [["word1"], ["word1", "w2"], ["word1", "w2", "wd3"]]

        char_ids = []

        for seq in sequences:
            ids = [[char_2_id_map.get(c, 0) for c in str(word)] for word in seq]
            char_ids.append(ids)

        sequence_padded, sequence_length = self.data_iterator._pad_sequences(sequences=char_ids,
                                                pad_tok=int(PAD_CHAR_ID),
                                                nlevels=2,
                                                MAX_WORD_LENGTH=5)
        np.testing.assert_array_equal(sequence_length, np.array([[5, 0, 0], [5, 2, 0], [5, 2, 3]]))
        expected = np.array([[[0, 1, 2, 3, 4],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]],

                             [[0, 1, 2, 3, 4],
                              [0, 5, 0, 0, 0],
                              [0, 0, 0, 0, 0]],

                             [[0, 1, 2, 3, 4],
                              [0, 5, 0, 0, 0],
                              [0, 3, 6, 0, 0]]])
        np.testing.assert_array_equal(sequence_padded, expected)
