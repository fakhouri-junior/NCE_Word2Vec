import os
import glob
from collections import Counter
import utils
import numpy as np
import random


OUTPUT_DIR = "vocab"

def build_the_vocab(dir_path, vocab_size, output_dir):
    """

    :param dir_path: directory where text files are stored to be used for building vocab
    :param vocab_size: size of the vocabulary to be constructed
    :return:
    """
    # create .tsv file with vocab_size
    utils.safe_mkdir(OUTPUT_DIR)
    output_file = open(os.path.join(output_dir, "vocab.tsv"), 'w', encoding="utf8")

    # read all the words
    all_words = []
    for txt_file in glob.glob(dir_path+"\\*.txt"):
        print(txt_file)
        words = open(txt_file, 'r', encoding="utf8").read()
        words = words.lower()
        words = ' '.join(words.split())

        words = words.replace('""'," ")
        words = words.replace(",", " ")
        words = words.replace("“", " ")
        words = words.replace("”", " ")
        words = words.replace(".", " ")
        words = words.replace(";", " ")
        words = words.replace("!", " ")
        words = words.replace("?", " ")
        words = words.replace("’", " ")
        words = words.replace("—", " ")

        words = words.split(' ')
        # check if empty words
        for word in words:
            if word:
                all_words.append(word)


    print("Number of words in all files is {}".format(len(all_words)))

    # Count all the words
    count = [('UNK', -1)]
    count.extend(Counter(all_words).most_common(vocab_size - 1))

    print("Number of unique words: {}".format(len(count)))
    print(count[:10])
    # write them to disk
    for word, _ in count:
        output_file.write(word + '\n')

    output_file.close()

    return os.path.join(output_dir, "vocab.tsv")

def get_dicts(vocab_file):
    word2int = {}
    int2word = {}
    all_words = open(vocab_file,'r').readlines()

    word2int = {word.strip(): index for index, word in enumerate(all_words)}
    int2word = {index: word.strip() for index, word in enumerate(all_words)}

    return word2int, int2word

def convert_words_to_indices(words, word2int):
    indices = []
    for word in words:
        if word in word2int:
            indices.append(word2int[word])
        else:
            indices.append(0) # UNK token is 0

    return indices

def read_data(txt_file):
    # important, parse the text files in the same way got parsed to generate vocab file
    words = open(txt_file, 'r', encoding="utf8").read()
    words = words.lower()
    words = ' '.join(words.split()) # get rid of many whitespaces

    words = words.replace('""'," ")
    words = words.replace(",", " ")
    words = words.replace("“", " ")
    words = words.replace("”", " ")
    words = words.replace(".", " ")
    words = words.replace(";", " ")
    words = words.replace("!", " ")
    words = words.replace("?", " ")
    words = words.replace("’", " ")
    words = words.replace("—", " ")

    words = words.split(' ')

    return [word.strip() for word in words]


def generate_data(text_files_directory, vocab_dir, vocab_size, context_window):
    # build the vocab
    vocab_file_path = build_the_vocab(text_files_directory, vocab_size, output_dir=vocab_dir)
    # get dictionaries of the vocab
    word2int, int2word = get_dicts(vocab_file_path)

    for txt_file in glob.glob(text_files_directory+"\\*.txt"):
        words = read_data(txt_file)
        txt_file_indexed = convert_words_to_indices(words, word2int)

        for index, center in enumerate(txt_file_indexed):
            context = random.randint(1, context_window)
            # get a random target before the center word
            # target has to be of shape [#, 1] , required by NCE
            for target in txt_file_indexed[max(0, index - context): index]:
                # print(np.expand_dims(np.array([target]), axis=1).shape)
                yield center, np.array([target])
            # get a random target after the center wrod
            for target in txt_file_indexed[index + 1: index + context + 1]:
                yield center, np.array([target])

#
# for testing purposes:
# my_generator = generate_data("data\\", "vocab\\", 10000, 4)
#
# for i in range(10):
#     print(next(my_generator))
