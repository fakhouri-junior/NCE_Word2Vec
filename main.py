import argparse
import os
import tensorflow as tf
import numpy as np

from tensorflow.contrib.training.python.training import hparam

import utils
import data_generator
import model

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

DATA_DIR = ""
VOCAB_DIR = ""
VOCAB_SIZE = None
CONEXT_SIZE = None
BATCH_SIZE = None

def gen():
    yield from data_generator.generate_data(DATA_DIR, VOCAB_DIR, VOCAB_SIZE, CONEXT_SIZE)


def build_dataset():
    dataset =  tf.data.Dataset.from_generator(gen, output_types=(tf.int32, tf.int32),
                            output_shapes=(tf.TensorShape([]), tf.TensorShape([1])))

    dataset = dataset.batch(BATCH_SIZE)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--context-window',
        default=10,
        help="Size of the window before and after the center words in skip gram",
        type=int
    )

    parser.add_argument(
        '--vocab-size',
        default=10000,
        help="Vocabulary size to be constructed and used",
        type=int
    )

    parser.add_argument(
        '--batch-size',
        default=128,
        type=int
    )

    parser.add_argument(
        '--embed-size',
        default=128,
        help="Dimensions of the embedded word",
        type=int
    )

    parser.add_argument(
        '--learning-rate',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--train-steps',
        default=500000,
        type=int
    )

    parser.add_argument(
        '--num-sampled',
        default=64,
        help="Number of words to sample in NCE loss",
        type=int
    )

    parser.add_argument(
        '--skip-step',
        default=5000,
        help="Every number of steps save the model and calculate average loss",
        type=int
    )

    parser.add_argument(
        '--num-visualize',
        default=5000,
        help="number of words to visualize from the vocab generated",
        type=int
    )

    parser.add_argument(
        '--data-dir',
        help="Directory where txt files are stored to be used for training and building vocab",
        type=str
    )

    parser.add_argument(
        '--vocab-dir',
        help="Directory where to store vocab generated file",
        type=str
    )

    args = parser.parse_args()

    DATA_DIR = args.data_dir
    VOCAB_DIR = args.vocab_dir
    CONEXT_SIZE = args.context_window
    VOCAB_SIZE = args.vocab_size
    BATCH_SIZE = args.batch_size

    dataset = build_dataset()
    # create the model
    word2vec = model.Word2Vec(dataset, args.context_window, args.vocab_size, args.batch_size,
                              args.embed_size, args.learning_rate, args.num_sampled,
                              args.skip_step, args.train_steps)

    word2vec.build_graph()
    word2vec.train()
    word2vec.visualize(args.vocab_dir, args.num_visualize)

