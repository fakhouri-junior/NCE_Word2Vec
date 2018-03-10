from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def most_common_words(vocab_dir, visual_fld, num_visualize):
    """ create a list of num_visualize most frequent words to visualize on TensorBoard.
    saved to visualization/vocab_[num_visualize].tsv
    """
    words = open(os.path.join(vocab_dir, 'vocab.tsv'), 'r').readlines()[:num_visualize]
    words = [word for word in words]
    safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab_' + str(num_visualize) + '.tsv'), 'w')
    for word in words:
        file.write(word)
    file.close()
