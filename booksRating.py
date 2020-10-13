import tensorflow as tf
import pickle
import numpy as np

"""
1.The books2idx dictionary is to get the vector data of the book according to the index:
#书ID转subscript dictionary, the book ID in the data set is inconsistent with the subscript, for example, the book ID in row 5 may not be 5
books2idx = {val[0]: i for i, val in enumerate(books.values)}
books2idx ---
{0: 0,
 1: 1,
 2: 2,
 ......}
    
2.books_isbn_mapThe dictionary is based on the real ISBN number to get the vector data of the book (recommended):
books_isbn_map ---
{'0195153448': 0,
 '0002005018': 1,
 '0060973129': 2,
 ......}
 
3.Vector data for this book:
[39193
 list([117609, 141801, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030, 106030])
 list([45850, 38173, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434, 19434])
 32
 list([5289, 4187, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624, 5624])]

The order is:
isbn	book_title	book_author	 year_of_publication	 publisher
"""
#This part is to forward the network and calculate the predicted score
def rating_books(new_model, user_id, isbn_id):
    #------------------------------------------------1.booksVector processing
    # 1.1 Get the index books_isbn_index corresponding to the isbn_id of the real book
    books_isbn_index = books_isbn_map[isbn_id]
    # 1.2 According to the index books_isbn_index, get the vector books_id_val of books
    books_val = books.values[books_isbn_index]
    # 1.3 Get the vector value of each field of the book
    isbn = books_val[0]
    year_of_publication = books_val[3]
    # Get a vector of book names
    book_title = np.zeros([1, 50])
    book_title[0] = books_val[1]
    # Get the vector of the book author
    book_author = np.zeros([1, 30])
    book_author[0] = books_val[2]
    # Get the vector of the book publisher
    publisher = np.zeros([1, 30])
    publisher[0] = books_val[4]

    # ------------------------------------------------2.usersVector processing
    # 2.1 Get the index user_id_index corresponding to the user_id of the real user
    user_id_index = user_id_map[user_id]
    # 2.2 According to the index user_id_index, get the user vector users_val
    users_val = users.values[user_id_index]
    # 2.3 The vector of user age obtained user_age
    user_age = users_val[1]

    #------------------------------------------------3.Load the model and predict the score
    inference_val = new_model([
        # User uid
        np.reshape(user_id, [1, 1]),

        # isbn_id
        np.reshape(isbn, [1, 1]),
        # book_title; List
        book_title,
        # book_author; List
        book_author,
        # year_of_publication
        np.reshape(year_of_publication, [1, 1]),
        # publisher; List
        publisher,

        # user_age；User's age
        np.reshape(user_age, [1, 1])])
    return (inference_val.numpy())

if __name__ == '__main__':
    #Load the preprocessed data;
    title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int, author2int, publisher2int, books_isbn_map, users_orig, books_orig = \
        pickle.load(open('booksPreprocess.p', mode='rb'))
    user_id_map = {val[0]: i for i, val in enumerate(users.values)}
    # --------------------------------------------------1.From the model that has been trained and saved in full; recreate the exact same model, including weights and optimizers
    new_model = tf.keras.models.load_model('./models/my_model.h5')
    #查看模型的参数
    new_model.summary()
    #指定用户和书籍进行评分
    """
   new_model ---loaded trained model
     user_id --- the user's real id; it is an int integer; no processing is required, it can be directly fed into the model for use
     isbn_id --- the real ISBN number of the book; string; need to find the corresponding index according to the dictionary books_isbn_map of the book;
    """
    user_id = 276798
    isbn_id = '349915398X'

    result = rating_books(new_model, user_id, isbn_id)[0][0]
    print('User {} rate the book{} is：{}'.format(user_id, isbn_id, result))

