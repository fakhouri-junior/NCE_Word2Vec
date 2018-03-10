import tensorflow as tf
import numpy as np
import model
import utils
import data_generator
import os

# when restoring variables , no need for initialization
embed_matrix = tf.get_variable("embed_matrix", shape=[10000, 128])

vocab_words = open("vocab\\vocab.tsv", 'r', encoding="utf8").readlines()
vocab_words = [word.strip() for word in vocab_words]


sess = tf.InteractiveSession()

def get_word_index(word):
    index = vocab_words.index(word)
    return index

def get_word_by_index(indices):
    for index in indices:
        print(vocab_words[index])


def restore_variable():

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints\\checkpoint"))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

def get_relation(two_positive_words, negative_word):
    # normalize
    nemb = tf.nn.l2_normalize(embed_matrix, 1)
    # get indices of the words
    indices = []
    for word in two_positive_words:
        indices.append(get_word_index(word))
    indices.append(get_word_index(negative_word))

    extracted_vectors = tf.gather(nemb, tf.convert_to_tensor(indices))
    res = (extracted_vectors[0] + extracted_vectors[1]) - extracted_vectors[2]

    dist = tf.matmul(nemb, tf.expand_dims(res,1))
    _, pred_idx = tf.nn.top_k(tf.transpose(dist), 4)
    my_list = sess.run(pred_idx)
    get_word_by_index(my_list[0])


if __name__ == "__main__":
    restore_variable()
    # get_relation(["north", "jaime"], "ned")
    """
    north river south jaime
    """
    # get_relation(["sword", "tyrion"], "jon")#sword blade dagger tyrion
    get_relation(["north", "daario"], "jon") # river, astapor
    # get_relation(["dragon", "jon"], "daenerys") # raven
    # get_relation(["north", "tywin"], "ned")  #south
    # get_relation(["cersei", "ned"], "joffrey")  #sansa
    # get_relation(["arya", "jon"], "sansa")  #sam theon
    # get_relation(["lannister", "jon"], "tywin")  #catelyn
    # get_relation(["jaime", "robb"], "tywin")  #ned
    # get_relation(["jaime", "dany"], "sword")  #tyrion lol
    get_relation(["tyrion", "dragon"], "dwarf")  #dany
    get_relation(["mother", "jon"], "joffrey")  #ghost
    get_relation(["strong", "cersei"], "dany")  # (dany - queen = cersei - what)




