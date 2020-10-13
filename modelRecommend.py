
import pickle
import tensorflow as tf
import numpy as np
import random

"""
1.Recommend books of the same type
The idea is to calculate the cosine similarity between the feature vector of the book currently viewed and the feature matrix of the entire book, and take the top_k with the largest similarity. Here, some random selections are added to ensure that each recommendation is slightly different.
"""
def recommend_same_type_books(books_id_val, top_k=20):
    #norm_books_matrics ---Standard Book Matrix
    norm_books_matrics = tf.sqrt(tf.reduce_sum(tf.square(books_matrics), 1, keepdims=True))
    #normalized_books_matrics ---Normalized book matrix
    normalized_books_matrics = books_matrics / norm_books_matrics

    # Recommend books of the same type
    probs_embeddings = (books_matrics[books_isbn_map[books_id_val]]).reshape([1, 200])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_books_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    print("The book you read is：{}".format(books_orig[books_isbn_map[books_id_val]]))
    print("The following are recommendations for you：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)

    results = set()
    while len(results) != top_k:
        """
        isbn_max ---266736
        np.random.choice ---Randomly extract elements from the array
            266736 ---Randomly extract numbers from the array np.arange(266736); form an array of the specified size (size);
            1 ---size 1
            p ---Array p: Corresponds to the array np.arange (266736), which means the probability of each element in the array np.arange (266736). The default is that the probability of selecting each element is the same.
                 p.shape=266736
        """
        c = np.random.choice(isbn_max, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(books_orig[val])
    return results

"""
2.Recommend your favorite books
The idea is to use the user feature vector and the book feature matrix to calculate the scores of all books, take the top_k with the highest score, and also add some random selection parts.
"""
def recommend_your_favorite_books(user_id_val, top_k=10):
    # Recommend books you like
    probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])

    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(books_matrics))
    sim = (probs_similarity.numpy())
    #     print(sim.shape)
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     sim_norm = probs_norm_similarity.eval()
    #     print((-sim_norm[0]).argsort()[0:top_k])

    print("The following are recommendations for you：")
    p = np.squeeze(sim)
    p[np.argsort(p)[:-top_k]] = 0
    p = p / np.sum(p)
    results = set()
    while len(results) != top_k:
        c = np.random.choice(isbn_max, 1, p=p)[0]
        results.add(c)
    for val in (results):
        print(books_orig[val])
    return results

"""
3.Those who have read this book also read (like) what books
First, select top_k individuals who like a certain book, and obtain the user feature vectors of these individuals.
Then calculate the ratings of these people for all books
Choose the highest-rated book for everyone as a recommendation
Random selection
"""
def recommend_other_favorite_books(books_id_val, top_k=20):
    probs_books_embeddings = (books_matrics[books_isbn_map[books_id_val]]).reshape([1, 200])
    probs_user_favorite_similarity = tf.matmul(probs_books_embeddings, tf.transpose(users_matrics))
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[0][-top_k:]
    #     print(normalized_users_matrics.numpy().shape)
    #     print(probs_user_favorite_similarity.numpy()[0][favorite_user_id])
    #     print(favorite_user_id.shape)

    print("The book you read is：{}".format(books_orig[books_isbn_map[books_id_val]]))

    print("People who like to read this book are：{}".format(users_orig[favorite_user_id - 1]))
    probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])
    probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(books_matrics))
    sim = (probs_similarity.numpy())
    #     results = (-sim[0]).argsort()[0:top_k]
    #     print(results)

    #     print(sim.shape)
    #     print(np.argmax(sim, 1))
    p = np.argmax(sim, 1)
    print("People who like to read this book also like to read：")

    if len(set(p)) < 5:
        results = set(p)
    else:
        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
    for val in (results):
        print(val)
        print(books_orig[val])
    return results

if __name__ == '__main__':
    #----------------------------------------------1.Load the preprocessed data;
    title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int, author2int, publisher2int, books_isbn_map, users_orig, books_orig = \
        pickle.load(open('booksPreprocess.p', mode='rb'))
    # ISBN  266736
    isbn_max = books.isbn.unique().max() + 1

    #----------------------------------------------2.Load the local user feature matrix users_matrics and book feature matrix
    books_matrics = pickle.load(open('books_matrics.p', mode='rb'))
    users_matrics = pickle.load(open('users_matrics.p', mode='rb'))

    #----------------------------------------------3.Start recommending books; use the produced user feature matrix users_matrics and book feature matrix for book recommendation
    #3.1.Recommend books of the same type
    """
    books_id_val ---ISBN number of the book  074322678X
    20 ---top20
    """
    recommend_same_type_books('074322678X', 20)

    #3.2.Recommend books you like
    """
    user_id_val ---User's ISBN number  234
    10 ---top10
    """
    recommend_your_favorite_books(234, 10)

    #3.3.Those who have read this book also read (like) what books
    """
    books_id_val ---ISBN number of the book  074322678X
    20 ---Those who have read this book, what other 20 books have they read
    """
    recommend_other_favorite_books('074322678X', 20)
