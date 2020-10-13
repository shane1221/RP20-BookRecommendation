
import pandas as pd
import pickle

"""
users_orig：Raw user data without data processing:
    array([[1, 'nyc, new york, usa', nan],
       [2, 'stockton, california, usa', 18.0],dtype=object)  
           
users：Pandas object of user data set
        	user_id	location	age
0	1	[5262, 19755, 20945, 17102, 11011, 11011, 1101...	0
1	2	[4561, 540, 17102, 11011, 11011, 11011, 11011,...	1

books_orig：Raw data without data processing:
    array([['0195153448', 'Classical Mythology', 'Mark P. O. Morford',
        2002.0, 'Oxford University Press'],
       ['0002005018', 'Clara Callan', 'Richard Bruce Wright', 2001.0,
        'HarperFlamingo Canada'],,dtype=object)
           
books：Pandas object of data
                isbn	book_title	book_author	year_of_publication	publisher
    0	0	[51322, 133154, 102065, 102065, 102065, 102065...	[37829, 49407, 4578, 13994, 16670, 16670, 1667...	0	[3774, 1946, 7827, 2813, 2813, 2813, 2813, 281...
    1	1	[42839, 118151, 102065, 102065, 102065, 102065...	[48223, 18963, 23253, 16670, 16670, 16670, 166...	1	[6648, 6804, 2813, 2813, 2813, 2813, 2813, 281...

books_users_ratings：Pandas object with three data sets combined together
        user_id	isbn	book_rating	book_title	book_author	year_of_publication	publisher	location	age
    0	276725	2918	0	[4382, 40930, 121308, 10279, 102065, 102065, 1...	[18514, 34135, 28773, 16670, 16670, 16670, 166...	0	[5412, 9056, 2813, 2813, 2813, 2813, 2813, 281...	[13252, 24704, 17102, 11011, 11011, 11011, 110...	0
    1	276726	221919	5	[7180, 49773, 32669, 102065, 102065, 102065, 1...	[5676, 31549, 16670, 16670, 16670, 16670, 1667...	1	[3064, 2813, 2813, 2813, 2813, 2813, 2813, 281...	[27820, 35917, 17102, 11011, 11011, 11011, 110...	0

features：Input X
    array([1, 1193, 0, 0, 10,
       list([4394, 2908, 3476, 3164, 4843, 46, 1224, 1224, 1224, 1224, 1224, 1224, 1224, 1224, 1224]),
       list([11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])],
      dtype=object)
      
--------------------------------------------------------------------------
targets_values：Is the learning target y, which is the predicted result
    array([[ 0],
           [ 5],
           ......])
       
book_ratings：Pandas object for scoring dataset
    user_id	isbn	book_rating
    0	276725	2918	0
    1	276726	221919	5

--------------------------------------------------------------------------

title_count：The length of the title（50）

title_set：Collection of book titles:
    {'Nobody',
     'Months',
     'Talented',
     'Bambi'......
    }     
"""
def load_data():
    #####################################################################################Import data set
    users = pd.read_csv('book_crossing_dataset/BX-Users.csv', sep=';')
    books = pd.read_csv('book_crossing_dataset/BX-Books.csv', sep=';')
    book_ratings = pd.read_csv('book_crossing_dataset/BX-Book-Ratings.csv', sep=';')

    #####################################################################################1.Processing user data
    # -----------------------------------------------------------1.1 User data cleaning
    #Organize column names
    users.columns = users.columns.str.strip().str.lower().str.replace('-', '_')
    # User age between 5-100 years old
    users = users[(users.age > 4) & (users.age < 101)]
    # users_orig The original user data contains the locatioon column; this is actually not used;
    users_orig = users.values
    #Users only need these 2 fields
    users = users[['user_id', 'age']]

    # -----------------------------------------------------------1.2 Vectorization of user data
    # ----------------1.2.1 Age field vectorization processing
    # Get the int mapping of age
    user_unique_age_set = users.age.unique()
    user_age_map = {o: i for i, o in enumerate(user_unique_age_set)}
    users['age'] = users['age'].map(user_age_map)

    ######################################################################################2.Process books data
    # -----------------------------------------------------------2.1 Books data cleaning
    # ---------------2.1.1 Load books data
    # Delete any line containing nan in books
    books = books.dropna(axis=0, how='any')

    # ---------------2.1.2 Organize column names. We can remove the image-url link column
    books.columns = books.columns.str.strip().str.lower().str.replace('-', '_')  # clean column names
    books.drop(columns=['image_url_s', 'image_url_m', 'image_url_l'], inplace=True)  # drop image-url columns

    # -----------------------------------------------------------2.2 Processing field publication yearyear_of_publication
    # Remove all the publication year year_of_publication as 0
    books = books[(books.year_of_publication != 0) & (books.year_of_publication != '0') & (
                books.year_of_publication != 'DK Publishing Inc') & (books.year_of_publication != 'Gallimard')]
    # Convert year to floating point
    books.year_of_publication = pd.to_numeric(books.year_of_publication, errors='coerce')
    #Create historical books and future books
    historical_books = books[books.year_of_publication < 1900]
    books_from_the_future = books[books.year_of_publication > 2020]
    #Delete historically published books and future published books
    books = books.loc[~(books.isbn.isin(historical_books.isbn))]
    books = books.loc[~(books.isbn.isin(books_from_the_future.isbn))]

    # -----------------------------------------------------------2.3 Processing field publisherPublisher
    #Clean up the ampersand format in the Publisher field
    books.publisher = books.publisher.str.replace('&amp', '&', regex=False)
    # books_orig Raw book data
    books_orig = books.values

    # -----------------------------------------------------------2.4 Unified vectorization
    #2.4.1 年份year_of_publication Vectorization
    year_map = {val: ii for ii, val in enumerate(books.year_of_publication.unique())}
    books['year_of_publication'] = books['year_of_publication'].map(year_map)

    #2.4.2 Vectorization of the publisher
    book_publisher_set = set()
    i = 0
    for val in books['publisher'].str.split():
        book_publisher_set.update(val)
        i += 1
    # If the length of the author name is less than 20, fill in <PAD>, and the index corresponding to <PAD> is 24208
    book_publisher_set.add('<PAD>')
    publisher2int = {val: ii for ii, val in enumerate(book_publisher_set)}
    # Convert the author name book_author into a list of equal length numbers, the length is 15
    publisher_count = 30
    publisher_map = {val: [publisher2int[row] for row in val.split()] for ii, val in enumerate(set(books['publisher']))}
    for key in publisher_map:
        for cnt in range(publisher_count - len(publisher_map[key])):
            publisher_map[key].insert(len(publisher_map[key]) + cnt, publisher2int['<PAD>'])
    books['publisher'] = books['publisher'].map(publisher_map)

    #2.4.3 book_authorVectorization of author name
    book_author_set = set()
    for val in books['book_author'].str.split():
        book_author_set.update(val)
    # If the length of the author name is less than 20, fill in <PAD>, and the index corresponding to <PAD> is 24208
    book_author_set.add('<PAD>')
    author2int = {val: ii for ii, val in enumerate(book_author_set)}
    # Convert the author name book_author into a list of equal length numbers, the length is 15
    author_count = 30
    author_map = {val: [author2int[row] for row in val.split()] for ii, val in enumerate(set(books['book_author']))}
    for key in author_map:
        for cnt in range(author_count - len(author_map[key])):
            author_map[key].insert(len(author_map[key]) + cnt, author2int['<PAD>'])
    books['book_author'] = books['book_author'].map(author_map)

    #######################################################################################3.Process book_ratings data
    book_ratings.columns = book_ratings.columns.str.strip().str.lower().str.replace('-', '_')
    #The score is actually from 1 to 10; but there are a lot of 0, 0 means a "implicit" rather than "explicit" evaluation; here the score of 0 is removed
    book_ratings = book_ratings[book_ratings.book_rating != 0]
    #Delete any line containing nan in book_ratings
    book_ratings = book_ratings.dropna(axis=0, how='any')

    ######################################################################################4.Connect the books table and book_ratings table using ISBN to get the books_with_ratings table
    books_with_ratings = book_ratings.join(books.set_index('isbn'), on='isbn')
    # Delete any rows containing nan in books_with_ratings
    books_with_ratings = books_with_ratings.dropna(axis=0, how='any')
    # After connecting to books_with_ratings, the year_of_publication field will automatically become float and manually change back to int
    books_with_ratings['year_of_publication'] = books_with_ratings['year_of_publication'].astype(int)

    #################################################5. At this time, in books, a book still corresponds to multiple ISBN codes, such as: Jane Eyre has many versions, e-books, wire-bound books, etc.; you must use ISBN to connect the two tables and book_ratings before removing the duplicates. , Otherwise it will delete some
    #Solve the problem that the same book may have multiple ISBN numbers (because it has different formats). We should clean it up before adding the'user' table
    multiple_isbns = books_with_ratings.groupby('book_title').isbn.nunique()
    has_mult_isbns = multiple_isbns.where(multiple_isbns > 1)
    # Delete nan, in this case it is a book with only one ISBN number
    has_mult_isbns.dropna(inplace=True)
    """
    Create the dictionary below and pickle it, just load it again (or run it for the first time on a new system).
    #Create a dictionary for books with multiple isbn
    """
    def make_isbn_dict(df):
        title_isbn_dict = {}
        for title in has_mult_isbns.index:
            print(title)
            isbn_series = df.loc[df.book_title==title].isbn.unique() # returns only the unique ISBNs
            title_isbn_dict[title] = isbn_series.tolist()
        return title_isbn_dict
    dict_unique_isbn = make_isbn_dict(books_with_ratings)
    #Since it takes a while for the loop to run (8 minutes on the full data set), pickle this data for future use
    with open('multiple_isbn_dict.pickle', 'wb') as handle:
        pickle.dump(dict_unique_isbn, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #I have trained'multiple_isbn_dict.pickle' before, now load isbn_dict back to the namespace
    with open('multiple_isbn_dict.pickle', 'rb') as handle:
        multiple_isbn_dict = pickle.load(handle)

    #Add the'unique_isbn' column to the'books_with_ratings' dataframe, including the first ISBN (if there are multiple ISBNs),
    def add_unique_isbn_col(df):
        df['unique_isbn'] = df.apply(lambda row: multiple_isbn_dict[row.book_title][
            0] if row.book_title in multiple_isbn_dict.keys() else row.isbn, axis=1)
        return df
    books_with_ratings = add_unique_isbn_col(books_with_ratings)

    #######################################################################6.Connect the books_with_ratings table and users table using user_id to get
    books_users_ratings = books_with_ratings.join(users.set_index('user_id'), on='user_id')
    # Delete any rows containing nan in users
    books_users_ratings = books_users_ratings.dropna(axis=0, how='any')
    # After connecting to books_with_ratings, the year_of_publication field will automatically become a float, here it is manually changed back to int
    books_users_ratings['age'] = books_users_ratings['age'].astype(int)
    books_users_ratings = books_users_ratings[
        ['user_id', 'unique_isbn', 'book_rating', 'book_title', 'book_author', 'year_of_publication', 'publisher',
         'age']]
    # Modify the column name, unique_isbn is changed to isbn
    books_users_ratings = books_users_ratings.rename(columns={'unique_isbn': 'isbn'})

    ####################################################################7.The two connection fields of user_id and isbn are also vectorized; user_id is already an int, no need to process
    #---------------------------------------------7.1 Vectorize the isbn field and book_title field of books
    # Vectorize the isbn field of books
    books_unique_isbn_set = books.isbn.unique()
    books_isbn_map = {o: i for i, o in enumerate(books_unique_isbn_set)}
    books['isbn'] = books['isbn'].map(books_isbn_map)

    # Vectorize the book_title field of books
    title_set = set()
    for val in books['book_title'].str.split():
        title_set.update(val)
    # If the length of the book name is less than 20, fill in <PAD>, the index corresponding to <PAD> is 24208
    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}
    # Turn book_title into a list of equal length numbers, the length is 15
    title_count = 50
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(books['book_title']))}
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])
    books['book_title'] = books['book_title'].map(title_map)

    # -------------------------7.2 Vectorize the isbn field and book_title field of books_users_ratings
    # Vectorize the isbn field of books_users_ratings
    books_users_ratings['isbn'] = books_users_ratings['isbn'].map(books_isbn_map)
    # Vectorize the book_title field of books_users_ratings
    books_users_ratings['book_title'] = books_users_ratings['book_title'].map(title_map)

    # -------------------------7.3 Vectorize the isbn field of book_ratings
    book_ratings['isbn'] = book_ratings['isbn'].map(books_isbn_map)
    # Delete the isbn field of the book_ratings table is empty
    book_ratings.dropna(subset=['isbn'], inplace=True)
    # The isbn field of the book_ratings table has been changed to float and converted to int
    book_ratings['isbn'] = book_ratings['isbn'].astype(int)

    #####################################################################8. Divide the data into two tables, X and y;
    # targets_pd == books_users_ratings[target_fields]  ------------y is the result we need to predict == the score of this book
    # features_pd == books_users_ratings.drop(target_fields, axis=1) -------------x is the data of users and books
    target_fields = ['book_rating']
    features_pd, targets_pd = books_users_ratings.drop(target_fields, axis=1), books_users_ratings[target_fields]
    features = features_pd.values
    targets_values = targets_pd.values
    return title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int,author2int,publisher2int,books_isbn_map,users_orig,books_orig

if __name__ == '__main__':
    title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int,author2int,publisher2int,books_isbn_map,users_orig,books_orig = load_data()
    # pickle.dump；Serialize the object and save the object obj to the file file
    pickle.dump((title_count, title_set, features, targets_values, book_ratings, users, books, books_users_ratings, title2int,author2int,publisher2int,books_isbn_map,users_orig,books_orig), open('booksPreprocess.p', 'wb'))
    print('The training set data initialization is complete......')


