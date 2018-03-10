import tensorflow as tf
import numpy as np
import os


class Word2vec_Eval:
    def __init__(self, vocab_filename_path="vocab\\vocab.tsv", embedding_matrix_variable_name="embed_matrix",
                 shape_of_matrix=[10000, 128],
                 checkpoint_path="checkpoints\\checkpoint"):

        self.embed_matrix = tf.get_variable(embedding_matrix_variable_name, shape=shape_of_matrix)
        self.checkpoint_path = checkpoint_path

        all_words = open(vocab_filename_path,'r', encoding="utf8").readlines()
        self.vocab_words = [word.strip() for word in all_words]
        self.sess = tf.Session()
        self.restore_embedding(self.sess)

    def restore_embedding(self, sess):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint_path))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Embedding Matrix Restored")

        print("Normalizing Embedding Matrix")
        self.embed_matrix = sess.run(tf.nn.l2_normalize(self.embed_matrix, 1))
        print("Done Normalizing")

    def get_index(self, word):
        if word in self.vocab_words:
            return self.vocab_words.index(word)
        else:
            print("word is not in vocab, UNK will be used")
            return 0 # 0 for UNK, unknown word

    def get_word(self, index):
        try:
            return self.vocab_words[index]
        except:
            print("out of range")
            return "UNK"

    def clean_up(self):
        self.sess.close()

    def most_common(self, word):
        # get word's index from vocab
        index = self.get_index(word)
        # get word vector
        word_vec = tf.gather(self.embed_matrix, tf.convert_to_tensor(index)) # the shape of word_vec will be =[embed_size] (shape_of_matrix[1])

        # get cosine distance with all vectors in embedding matrix
        dist = tf.matmul(tf.expand_dims(word_vec, 0), tf.transpose(self.embed_matrix))
        nearby_val, nearby_idx = tf.nn.top_k(dist, 10)

        indices = self.sess.run(nearby_idx)
        res = []
        for j in indices:
            for i in j:
                res.append(self.get_word(i))

        print(res)

    def analogy(self,positive_words, negative_word):
        # get indices for words
        indices = []
        for word in positive_words:
            indices.append(self.get_index(word))
        indices.append(self.get_index(negative_word))

        # extract vectors
        ang_vec = tf.gather(self.embed_matrix, tf.convert_to_tensor(indices))
        x = (ang_vec[0] + ang_vec[1]) - ang_vec[2]
        dist = tf.matmul(tf.expand_dims(x,0), np.transpose(self.embed_matrix))
        nearby_val, nearby_ids = tf.nn.top_k(dist, 4)

        indices = self.sess.run(nearby_ids)
        res = []
        for j in indices:
            for i in j:
                res.append(self.get_word(i))

        print(res)

if __name__ == "__main__":
    word2vec_eval = Word2vec_Eval(vocab_filename_path="vocab\\vocab.tsv",
                                  embedding_matrix_variable_name="embed_matrix",
                                  shape_of_matrix=[10000,128],
                                  checkpoint_path="checkpoints\\checkpoint")
    word2vec_eval.most_common("dragon")
    word2vec_eval.analogy(positive_words=["mother", "jon"], negative_word="joffrey")
