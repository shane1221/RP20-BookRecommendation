import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import os
import pickle
import re
from tensorflow.python.ops import math_ops
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time

# -------------------------------------------3.Input; define a placeholder for input
def get_inputs():
    print('---------------------------------2. get_inputs Get input data')
    # uid User ID
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')
    # user_age User age
    user_age = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_age')

    # isbn_id Book id
    isbn_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='isbn_id')
    # year_of_publication_id Book publication year
    year_of_publication = tf.keras.layers.Input(shape=(1,), dtype='int32', name='year_of_publication_id')
    # book_title Book title; list, length 20
    book_title = tf.keras.layers.Input(shape=(50,), dtype='int32', name='book_title')
    # book_author Book authors; list, length 15
    book_author = tf.keras.layers.Input(shape=(30,), dtype='int32', name='book_author')
    # publisher Book publisher; list, length 15
    publisher = tf.keras.layers.Input(shape=(30,), dtype='int32', name='publisher')
    return uid, isbn_id, book_title, book_author, year_of_publication, publisher, user_age

# batch
def get_batches(Xs, ys, batch_size):
    print('---------------------------------12. get_batches---Get batch')
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

# -------------------------------------------4.Building a neural network
# 4.1 Define the embedding matrix of User
def get_user_embedding(uid, user_age):
    print('---------------------------------3. get_user_embedding User Embedding matrix')
    """
    uid_max: dictionary size
     embed_dim: The output size of this layer, which is the dimension of the generated embedding
     input_length: The dimension of the input data. Because the input data will be padding processed, it is generally the defined max_length
    """
    # uid  ---Tensor("uid:0", shape=(None, 1), dtype=int32)
    uid_embed_layer = tf.keras.layers.Embedding(uid_max, embed_dim, input_length=1, name='uid_embed_layer')(uid)
    print('uid..........................', uid)
    print('user_age..........................', user_age)
    print('embed_dim // 2..........................', embed_dim // 2)
    # age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim // 2, input_length=1, name='age_embed_layer')(tf.cast(user_age, tf.int32))
    age_embed_layer = tf.keras.layers.Embedding(age_max, embed_dim // 2, input_length=1, name='age_embed_layer')(
        user_age)
    return uid_embed_layer, age_embed_layer

# Fully connect User's embedded matrix together to generate User characteristics
def get_user_feature_layer(uid_embed_layer, age_embed_layer):
    print('---------------------------------4. get_user_feature_layer User embedding matrix is fully connected together to generate User features')
    # First layer fully connected
    uid_fc_layer = tf.keras.layers.Dense(embed_dim, name="uid_fc_layer", activation='relu')(uid_embed_layer)
    age_fc_layer = tf.keras.layers.Dense(embed_dim, name="age_fc_layer", activation='relu')(age_embed_layer)

    # The second layer is fully connected
    user_combine_layer = tf.keras.layers.concatenate([uid_fc_layer, age_fc_layer], 2)  # (?, 1, 128)
    user_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(user_combine_layer)  # (?, 1, 200)

    user_combine_layer_flat = tf.keras.layers.Reshape([200], name="user_combine_layer_flat")(user_combine_layer)
    return user_combine_layer, user_combine_layer_flat

# 4.2 Define the embedding matrix of books
# Embedding vector of isbn
def get_isbn_embed_layer(isbn):
    print('---------------------------------5. get_isbn_embed_layer---Get isbn Embedding vector')
    isbn_embed_layer = tf.keras.layers.Embedding(isbn_max, embed_dim, input_length=1, name='isbn_embed_layer')(isbn)
    return isbn_embed_layer

# Embedding vector of year_of_publication
def get_year_of_publication_embed_layer(year_of_publication):
    print('---------------------------------6. get_year_of_publication_embed_layer---Get year_of_publication Embedding vector')
    year_of_publication_embed_layer = tf.keras.layers.Embedding(year_of_publication_max, embed_dim, input_length=1,
                                                                name='year_of_publication_embed_layer')(
        year_of_publication)
    return year_of_publication_embed_layer

# Combine multiple embedding vectors of book_author
def get_book_author_layers(book_author):
    print('---------------------------------7.get_book_author_layers merge book_author Embedding vector')
    book_author_embed_layer = tf.keras.layers.Embedding(author_max, embed_dim, input_length=30,
                                                        name='book_author_embed_layer')(book_author)
    book_author_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        book_author_embed_layer)
    return book_author_embed_layer

# Merge multiple embedding vectors of publisher
def get_publisher_layers(publisher):
    print('---------------------------------8. get_publisher_layers merge publisher Multiple embedding vectors')
    publisher_embed_layer = tf.keras.layers.Embedding(publisher_max, embed_dim, input_length=18,
                                                      name='publisher_embed_layer')(publisher)
    publisher_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        publisher_embed_layer)
    return publisher_embed_layer

# Book_title's text convolutional network realizes feature extraction
def get_books_cnn_layer(book_title):
    print('---------------------------------9. get_books_cnn_layer---book_title Implementation of text convolutional network')
    # Get the embedding vector of each word corresponding to the book name from the embedding matrix; input_length=50 This is the maximum length of the book name
    book_title_embed_layer = tf.keras.layers.Embedding(title_max, embed_dim, input_length=50,
                                                       name='book_title_embed_layer')(book_title)
    # sp........................(None, 20, 32)
    sp = book_title_embed_layer.shape
    book_title_embed_layer_expand = tf.keras.layers.Reshape([sp[1], sp[2], 1])(book_title_embed_layer)
    # Use convolution kernels of different sizes for the text embedding layer for convolution and maximum pooling
    pool_layer_lst = []
    # window_sizes = {2, 3, 4, 5}
    for window_size in window_sizes:
        """
        tf.keras.layers.Conv2D ---Add convolutional layer
         filter_num: the number of text convolution kernels (that is, the dimension of the output) ---8
         kernel_size: integer or list/tuple composed of a single integer, the spatial or time domain window length of the convolution kernel ---(2, 32) and (3, 32) and (4, 32) and (5, 32)
         strides: an integer or a list/tuple composed of a single integer, which is the step size of the convolution. Any strides that is not 1 are incompatible with any dilation_rate that is not 1 ---1
         activation: activation function ---'relu'
        """
        conv_layer = tf.keras.layers.Conv2D(filter_num, (window_size, embed_dim), 1, activation='relu')(
            book_title_embed_layer_expand)
        # tf.keras.layers.MaxPooling2D ---Add maximum pooling layer
        maxpool_layer = tf.keras.layers.MaxPooling2D(pool_size=(sentences_size - window_size + 1, 1), strides=1)(
            conv_layer)
        pool_layer_lst.append(maxpool_layer)

    # tf.keras.layers.concatenate ---axis=3,Indicates splicing from the third dimension
    pool_layer = tf.keras.layers.concatenate(pool_layer_lst, 3, name="pool_layer")
    max_num = len(window_sizes) * filter_num
    pool_layer_flat = tf.keras.layers.Reshape([1, max_num], name="pool_layer_flat")(pool_layer)
    # Add the Dropout layer; apply Dropout to the input data. Dropout will randomly disconnect input neurons at a certain probability (rate) each time the parameters are updated during the training process, and the Dropout layer is used to prevent overfitting.
    dropout_layer = tf.keras.layers.Dropout(dropout_keep, name="dropout_layer")(pool_layer_flat)
    return pool_layer_flat, dropout_layer

# Connect all layers of books together
def get_books_feature_layer(isbn_embed_layer, year_of_publication_embed_layer, book_author_embed_layer,
                            publisher_embed_layer,
                            dropout_layer):
    print('---------------------------------10. get_books_feature_layer Connect all layers of books together')
    # The first layer is fully connected; each feature of the book (ISBN number, publication year, author, publisher, etc.) is transferred to the fully connected layer
    isbn_fc_layer = tf.keras.layers.Dense(embed_dim, name="isbn_fc_layer", activation='relu')(isbn_embed_layer)
    year_of_publication_fc_layer = tf.keras.layers.Dense(embed_dim, name="year_of_publication_categories_fc_layer",
                                                         activation='relu')(year_of_publication_embed_layer)
    book_author_fc_layer = tf.keras.layers.Dense(embed_dim, name="book_author_fc_layer", activation='relu')(
        book_author_embed_layer)
    publisher_fc_layer = tf.keras.layers.Dense(embed_dim, name="publisher_fc_layer", activation='relu')(
        publisher_embed_layer)

    """
    #input_length=1
    isbn_embed_layer = tf.keras.layers.Embedding(isbn_max, embed_dim, input_length=1, name='isbn_embed_layer')(isbn)
    #embed_dim=32
    isbn_fc_layer = tf.keras.layers.Dense(embed_dim, name="isbn_fc_layer", activation='relu')(isbn_embed_layer)
    isbn_fc_layer  shape=(None, 1, 32);
    -------------------------------------------------
    isbn_fc_layer ---<tf.Tensor 'isbn_fc_layer_2/Identity:0' shape=(None, 1, 32) dtype=float32>
    year_of_publication_fc_layer ---<tf.Tensor 'year_of_publication_categories_fc_layer_2/Identity:0' shape=(None, 1, 32) dtype=float32>
    book_author_fc_layer ---<tf.Tensor 'book_author_fc_layer_2/Identity:0' shape=(None, 1, 32) dtype=float32>
    publisher_fc_layer ---<tf.Tensor 'publisher_fc_layer_2/Identity:0' shape=(None, 1, 32) dtype=float32>
    dropout_layer ---<tf.Tensor 'dropout_layer_2/Identity_1:0' shape=(None, 1, 32) dtype=float32>
    books_combine_layer  ---<tf.Tensor 'concatenate_8/Identity:0' shape=(None, 1, 160) dtype=float32>；
    """
    books_combine_layer = tf.keras.layers.concatenate(
        [isbn_fc_layer, year_of_publication_fc_layer, book_author_fc_layer, publisher_fc_layer, dropout_layer], 2)
    books_combine_layer = tf.keras.layers.Dense(200, activation='tanh')(books_combine_layer)

    books_combine_layer_flat = tf.keras.layers.Reshape([200], name="books_combine_layer_flat")(books_combine_layer)
    return books_combine_layer, books_combine_layer_flat

# Build a neural network calculation graph
class mv_network(object):
    def __init__(self, batch_size=256):
        print('---------------------------------1.Build a neural network calculation graph')
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}
        # -------------------------1.1 Get input placeholder
        uid, isbn_id, book_title, book_author, year_of_publication, publisher, user_age = get_inputs()

        # --------------------------1.2 Get the user's feature vector user_combine_layer_flat
        # 2.1 get_user_embedding ---Get 3 embedding vectors of User
        uid_embed_layer, age_embed_layer = get_user_embedding(uid, user_age)
        # 2.2 get_user_feature_layer ---Fully connected layer twice to get user characteristics
        user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, age_embed_layer)

        # --------------------------1.3 Get the feature vector of books books_combine_layer_flat
        # get_isbn_embed_layer ---Get the embedding vector of isbn
        isbn_embed_layer = get_isbn_embed_layer(isbn_id)
        # get_year_of_publication_embed_layer ---Get the embedding vector of year_of_publication_id
        year_of_publication_embed_layer = get_year_of_publication_embed_layer(year_of_publication)
        # get_book_author_layers ---Get the embedding vector of isbn
        book_author_embed_layer = get_book_author_layers(book_author)
        # get_publisher_layers ---Get the embedding vector of isbn
        publisher_embed_layer = get_publisher_layers(publisher)
        # get_books_cnn_layer ---Use CNN neural network to get the feature vector of book name
        pool_layer_flat, dropout_layer = get_books_cnn_layer(book_title)
        # get_books_feature_layer ---Get the book characteristics and do 2 full connection layers
        books_combine_layer, books_combine_layer_flat = get_books_feature_layer(isbn_embed_layer,
                                                                                year_of_publication_embed_layer,
                                                                                book_author_embed_layer,
                                                                                publisher_embed_layer,
                                                                                dropout_layer)

        # ------------------------------------------------------------4.Method one: Do matrix multiplication of user characteristics and book characteristics to obtain predicted scores; regress the predicted scores with the real scores and use MSE to optimize the loss. Because this is essentially a regression problem; (The MSE loss after 5 iterations in this way is around 1)
        # -----4.1 Do matrix multiplication of user features and book features to get the predicted score inference
        inference = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer[0] * layer[1], axis=1), name="inference")(
            (user_combine_layer_flat, books_combine_layer_flat))
        inference = tf.keras.layers.Lambda(lambda layer: tf.expand_dims(layer, axis=1))(inference)

        # -----4.2 Regress the predicted score with the real score, and use MSE (Mean Square Error) to optimize the loss; MSE (Mean Square Error) is the average of the sum of squares of the difference between the predicted value and the true value
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        # tf.keras.metrics.MeanAbsoluteError() ---Calculate the average absolute error between the label and the prediction
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()

        # -----4.3 定义模型model和优化器optimizers
        # tf.keras.Model类将定义好的网络结构封装入一个对象，用于训练、测试和预测
        """
        模型的训练参数的顺序是：
         uid, isbn_id, book_title, book_author, year_of_publication, publisher, user_age
        """
        self.model = tf.keras.Model(
            inputs=[uid, isbn_id, book_title, book_author, year_of_publication, publisher, user_age],
            outputs=[inference])
        # Output the parameter status of each layer of the model through model.summary()
        print('The parameter status of each layer of the output model through model.summary() is as follows：', self.model.summary())
        """
        The optimizer keras.optimizers.Adam()---adopts dynamic learning rate decay to solve the problem of inability to quickly fit the model due to excessive learning rate and low training efficiency due to low learning rate;
         The general idea is that the initial learning rate is set to a larger value, and then according to the increase in the number of times, the learning rate is dynamically reduced to achieve both efficiency and effect.
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        # If the "./models" folder does not exist, create a new one
        if tf.io.gfile.exists(MODEL_DIR):
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        # -----4.4 Set up checkpoints during initialization;
        # checkpoint_dir  ---  ./models/checkpoints
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        # checkpoint_prefix --- ./models/checkpoints/ckpt
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_loss(self, labels, logits):
        print('---------------------------------compute_loss')
        return tf.reduce_mean(tf.keras.losses.mse(labels, logits))

    def compute_metrics(self, labels, logits):
        print('---------------------------------compute_metrics')
        return tf.keras.metrics.mae(labels, logits)  #

    @tf.function
    def train_step(self, x, y):
        print('---------------------------------13. train_step')
        # ----------------------------------------1.tf.GradientTape梯度求解利器，采用MSE(均方误差)算法；得到梯度grads
        # tf.GradientTape梯度求解利器；GradientTape会监控可训练变量;这样即可计算出所有可训练变量的梯度，然后进行下一步的更新;
        with tf.GradientTape() as tape:

            print('train_step............：', x)
            print('train_step............：', y)
            logits = self.model([x[0],
                                 x[1],
                                 x[2],
                                 x[3],
                                 x[4],
                                 x[5],
                                 x[6]], training=True)
            print('1111111111111111111111111111111111')
            loss = self.ComputeLoss(y, logits)

            print('2222222222222222222222222222222222222222222')
            self.ComputeMetrics(y, logits)
            print('33333333333333333333333333333')
        grads = tape.gradient(loss, self.model.trainable_variables)
        print('44444444444444444444444444444444')
        # ----------------------------------------2.Update self.model.trainable_variables with gradient grads as input

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        print('555555555555555555555555555555555555')
        print('loss..................',loss)
        print('logits..................',logits)
        return loss, logits

    def training(self, features, targets_values, epochs=5, log_freq=50):
        print('---------------------------------11. training')
        # ---------------------1.Cycle training for the specified number of epochs; after each round of training is completed, test testing is performed automatically, and the callback point is stored when the loss is smaller than before
        for epoch_i in range(epochs):

            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            if True:
                start = time.time()

                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

                for batch_i in range(batch_num):

                    x, y = next(train_batches)
                    print('batch_i...........................x',len(x))
                    print('batch_i...........................y',len(y))


                    book_title = np.zeros([self.batch_size, 50])
                    for i in range(self.batch_size):
                        book_title[i] = x.take(2, 1)[i]

                    book_author = np.zeros([self.batch_size, 30])
                    for i in range(self.batch_size):
                        book_author[i] = x.take(3, 1)[i]

                    publisher = np.zeros([self.batch_size, 30])
                    for i in range(self.batch_size):
                        publisher[i] = x.take(5, 1)[i]

                    """
                    x:
                    uid, isbn_id, book_title, book_author, year_of_publication, publisher, user_age

                    [198711 23728
                     list([94074, 51982, 98088, 105208, 108635, 108117, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090, 107090])
                     list([12688, 45680, 3223, 20197, 20197, 20197, 20197, 20197, 20197, 20197, 20197, 20197, 20197, 20197, 20197])
                     15
                     list([3432, 9710, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228, 6228])
                     38]
                    """
                    print('#######################################################################1')

                    loss, logits = self.train_step([
                        # uid
                        np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                        # isbn_id
                        np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                        # book_title;
                        book_title.astype(np.float32),
                        # book_author;
                        book_author.astype(np.float32),
                        # year_of_publication
                        np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                        # publisher;
                        publisher.astype(np.float32),
                        # user_age
                        np.reshape(x.take(6, 1), [self.batch_size, 1]).astype(np.float32)],
                        np.reshape(y, [self.batch_size, 1]).astype(np.float32))

                    print('#######################################################################2')
                    print('####################################################################',loss, logits)
                    avg_loss(loss)
                    self.losses['train'].append(loss)
                    # log_freq---50；
                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))

                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        start = time.time()

            train_end = time.time()
            print(
                '\nThe {} epoch ######({}steps)######Time： {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                        train_end - train_start))

            self.testing((test_X, test_y), self.optimizer.iterations)
        # ---------------------2.After the training is completed, store the trained model in the'export' folder;
        # self.export_path = os.path.join(MODEL_DIR, 'export')
        # tf.saved_model.save(self.model, self.export_path)
        self.model.save(MODEL_DIR + '/my_model.h5')
        print('After training, save the complete model parameters..........................')

    def testing(self, test_dataset, step_num):
        print('---------------------------------14. testing')
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)


        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)


            # book_title.shape()。。。。。。。。。。。。 (256, 25)
            book_title = np.zeros([self.batch_size, 50])
            for i in range(self.batch_size):
                book_title[i] = x.take(2, 1)[i]

            book_author = np.zeros([self.batch_size, 30])
            for i in range(self.batch_size):
                book_author[i] = x.take(3, 1)[i]

            publisher = np.zeros([self.batch_size, 30])
            for i in range(self.batch_size):
                publisher[i] = x.take(5, 1)[i]

            logits = self.model([  # uid
                np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                # isbn_id
                np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                # book_title;
                book_title.astype(np.float32),
                # book_author;
                book_author.astype(np.float32),
                # year_of_publication
                np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                # publisher;
                publisher.astype(np.float32),
                # user_age
                np.reshape(x.take(6, 1), [self.batch_size, 1]).astype(np.float32)], training=False)

            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
            avg_loss(test_loss)
            # Save test loss
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), logits)
        print('The loss of the model test is: {:0.6f} ###### The calculated index mae is: {:0.6f}'.format(avg_loss.result(), self.ComputeMetrics.result()))
        # If the test finds that the loss of the model trained in this round is smaller, save it;
        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            print('================This round of loss is smaller, savecheckpoint===================')
            self.checkpoint.save(self.checkpoint_prefix)

    # forward
    def forward(self, xs):
        print('---------------------------------forward')
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)
        return predictions

if __name__ == '__main__':

    """
    features  ---
    user_id,	isbn,	book_title,	book_author,	year_of_publication,	publisher,	age

    array([276725, 2918,
       list([129120, 48637, 58489, 1987, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982, 83982]),
       list([51394, 41631, 25913, 27218, 27218, 27218, 27218, 27218, 27218, 27218, 27218, 27218, 27218, 27218, 27218]),
       0,
       list([5908, 8630, 268, 268, 268, 268, 268, 268, 268, 268, 268, 268, 268, 268, 268]),
       0], dtype=object)
    """
    title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int,author2int,publisher2int,books_isbn_map,users_orig,books_orig = \
        pickle.load(open('booksPreprocess0917.p', mode='rb'))

    # -------------------------------------------1.Common parameter setting
    # -------------------------1.1 User parameters
    # Number of user IDs 278856
    uid_max = users.user_id.unique().max() + 1

    # age  96
    age_max = users.age.unique().max() + 1

    # -------------------------1.2 Parameters of books
    # ISBN 266736
    isbn_max = books.isbn.unique().max() + 1

    # year_of_publication   104
    year_of_publication_max = books.year_of_publication.unique().max() + 1

    # book_title 155718
    title_max = max(title2int.values()) + 1
    # book_title 155718
    # title_max = len(title_set)

    # book_author  53768
    author_max = max(author2int.values()) + 1
    # publisher  12324
    publisher_max = max(publisher2int.values()) + 1

    # booksid2idx = {val[0]: i for i, val in enumerate(books.values)}

    # -------------------------1.3 Model training parameters
    # The storage path of the trained model
    MODEL_DIR = "./models"
    save_dir = './save'
    # Embedding matrix dimensions
    embed_dim = 32
    # Title length 50
    sentences_size = title_count
    # Text convolution sliding window, sliding 2, 3, 4, 5 words respectively
    window_sizes = {2, 3, 4, 5}
    # Number of text convolution kernels
    filter_num = 8

    # -------------------------------------------2.Hyper parameter settings
    # Number of rounds of model training
    num_epochs = 1
    # Training 256 pieces of data per batch
    batch_size = 256
    dropout_keep = 0.5
    # Learning rate
    learning_rate = 0.0001
    # -------------------------------------------3.Training model
    # Train the network; take user characteristics and book characteristics as input, and then output a value through full connection
    mv_net = mv_network()
    mv_net.training(features, targets_values, epochs=1)
    print('###############################The score prediction model is trained######################################')