import tensorflow as tf
import numpy as np
import utils
from tensorflow.contrib.tensorboard.plugins import projector
import os

VIS_FLD = "visualizations"

class Word2Vec:
    def __init__(self, dataset, context_window, vocab_size, batch_size, embed_size, learning_rate,
                 num_sampled, skip_step, train_steps):

        self.dataset = dataset
        self.train_steps = train_steps
        self.context_window = context_window
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.num_sampled = num_sampled
        self.skip_step = skip_step

        self.global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)

    def _import_data(self):
        with tf.name_scope("data"):
            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)

            self.center_words, self.target_words = self.iterator.get_next()

            self.init_iterator_train = self.iterator.make_initializer(self.dataset)

    def _create_embedding(self):
        with tf.name_scope("embed"):
            self.embed_matrix = tf.get_variable("embed_matrix", shape=[self.vocab_size, self.embed_size],
                                            initializer=tf.random_uniform_initializer())
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name="embedding")

            print(self.embed)
    def _create_loss(self):
        with tf.name_scope('loss'):
            nce_weight = tf.get_variable("nce_weight", shape=[self.vocab_size, self.embed_size],
                                         initializer=tf.truncated_normal_initializer(stddev=1.0 / (self.embed_size ** 0.5) ))

            nce_bias = tf.get_variable("nce_bias", initializer=tf.zeros([self.vocab_size]))

            print(self.target_words)
            print(self.center_words)

            self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=self.target_words,
                                                      inputs=self.embed, num_sampled=self.num_sampled,
                                                      num_classes=self.vocab_size), name="loss")

    def _create_optimizier(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)


    def _create_summaries(self):
        with tf.name_scope("summary"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._import_data()
        self._create_embedding()
        self._create_loss()
        self._create_optimizier()
        self._create_summaries()

    def train(self):
        init_op = tf.global_variables_initializer()
        utils.safe_mkdir("graph\\")
        writer = tf.summary.FileWriter("graph\\", tf.get_default_graph())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # initialize variables
            sess.run(init_op)
            # initialize dataset
            sess.run(self.init_iterator_train)
            total_loss = 0.0

            # restore if any
            utils.safe_mkdir("checkpoints\\")
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints\\checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            initial_step = self.global_step.eval()

            for index in range(initial_step, initial_step + self.train_steps):
                try:
                    loss, _, summary = sess.run([self.loss, self.optimizer, self.summary_op])
                    # add summary to filewriter
                    writer.add_summary(summary, global_step=index)

                    total_loss += loss
                    if (index + 1) % self.skip_step == 0:
                        print("Average loss at step {}: {:5.1f}".format(index, total_loss / self.skip_step))
                        total_loss = 0.0
                        saver.save(sess, "checkpoints\\word2vec", global_step=index)

                except tf.errors.OutOfRangeError:
                    sess.run(self.init_iterator_train)

        writer.close()

    def visualize(self, vocab_dir, num_visualize):
        # create a file containing num_visualize words to be used for visualization
        utils.most_common_words(vocab_dir, VIS_FLD, num_visualize)

        # access embed_matrix variable generated during training to extract the first num_visualize from it
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints\\checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("Can't find checkpoint path, errgo can't visualzie")
                return

            saved_embed_matrix = sess.run(self.embed_matrix)

            embedded_matrix = tf.Variable(saved_embed_matrix[:num_visualize], name="embedded_matrix_visualize")
            sess.run(embedded_matrix.initializer)

            config = projector.ProjectorConfig()
            file_writer = tf.summary.FileWriter(VIS_FLD)

            embedding = config.embeddings.add()
            embedding.tensor_name = embedded_matrix.name
            embedding.metadata_path = os.path.join(VIS_FLD, "vocab_"+str(num_visualize)+".tsv")

            projector.visualize_embeddings(file_writer, config)
            # save the model with variable to be read by tensorboard in vis_folder
            embed_saver = tf.train.Saver([embedded_matrix])
            embed_saver.save(sess, os.path.join(VIS_FLD, 'model.ckpt'), 1)









