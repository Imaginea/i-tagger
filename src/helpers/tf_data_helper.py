from tensorflow.python.platform import gfile
import tensorflow as tf
from tqdm import tqdm

def tf_vocab_processor(lines, out_file_name, max_doc_length=1000, min_frequency=0):
    # Create vocabulary
    # min_frequency -> consider a word if and only it repeats for fiven count
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_doc_length,
                                                                         min_frequency=min_frequency)
    vocab_processor.fit(lines)

    vocab = vocab_processor.vocabulary_._mapping.keys()

    # Create a file and store the words
    with gfile.Open(out_file_name, 'wb') as f:
        for word in vocab:
            f.write("{}\n".format(word))

    nwords =len(vocab_processor.vocabulary_) + 1 #<UNK>

    print('{} words into {}'.format(nwords, out_file_name))
    return (nwords, vocab)

def get_sequence_length_old(sequence):
    '''
    Returns the sequence length, droping out all the zeros if the sequence is padded
    :param sequence: Tensor(shape=[batch_size, doc_length, feature_dim])
    :return: Array of Document lengths of size batch_size
    '''
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used,1)
    length = tf.cast(length, tf.int32)
    return length


def get_sequence_length(sequence_ids, pad_word_id=0):
    '''
    Returns the sequence length, droping out all the padded tokens if the sequence is padded

    :param sequence_ids: Tensor(shape=[batch_size, doc_length])
    :param pad_word_id: 0 is default
    :return: Array of Document lengths of size batch_size
    '''
    flag = tf.greater_equal(sequence_ids, 1) # TODO 1 -> start of <UNK> vocab id
    used = tf.cast(flag, tf.int32)
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels, max_doc_length, max_word_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length  = max_doc_length
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)
        #breaking the code to pad the string instead on its ids

        # print_info(sequence_length)
    elif nlevels == 2:
        # max_length_word = max([max(map(lambda x: len(x), seq))
        #                        for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in tqdm(sequences, desc="pad_sequences"):
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_word_length)
            sequence_padded += [sp]
            sequence_length += [sl]

        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_word_length,
                                            max_doc_length)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_doc_length)

    return sequence_padded, sequence_length