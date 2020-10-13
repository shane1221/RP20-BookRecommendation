# RP20-BookRecommendation

python3.6.5 or anaconda3 5.2.0
tensorflow2：
pip3 install -i https://pypi.doubanio.com/simple/ tensorflow-cpu==2.2.0

----------------------------------------------------------



bookDataPreprocess.pyTraining Data preprocessing program：

1.Read training dataBX-Users.csv、BX-Books.csv、BX-Book-Ratings.csv

2.Clean the data, and complete the data vectorization; return：
users_orig：Raw user data without data processing:
    array([[1, 'nyc, new york, usa', nan],
       [2, 'stockton, california, usa', 18.0],dtype=object)  
users：Vector of user data set：
        	user_id	location	age
0	1	[5262, 19755, 20945, 17102, 11011, 11011, 1101...	0
1	2	[4561, 540, 17102, 11011, 11011, 11011, 11011,...	1

books_orig：Original book data without data processing:
    array([['0195153448', 'Classical Mythology', 'Mark P. O. Morford',
        2002.0, 'Oxford University Press'],
       ['0002005018', 'Clara Callan', 'Richard Bruce Wright', 2001.0,
        'HarperFlamingo Canada'],,dtype=object)
           
books：Vector of book data：
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
      
targets_values：Is the learning target y, which is the predicted result
    array([[ 0],
           [ 5],
           ......])
       
book_ratings：Pandas object for scoring dataset
    user_id	isbn	book_rating
    0	276725	2918	0
    1	276726	221919	5

title_count：The length of the title (50)

title_set：Collection of book titles:
    {'Nobody',
     'Months',
     'Talented',
     'Bambi'......
    }    
    
3.The processing result is serialized and saved to the local booksPreprocess.p file

----------------------------------------------------------------------------------------------------------------

modelTrain.pyModel training file:
1.Load the processed training data from the booksPreprocess0912.p file

2.Construct the embedding layer Embedding and the fully connected layer Dense to extract user features and book features (where the book name uses a text convolutional network to achieve feature extraction).

3.tf.keras.ModelBuild a model; do a matrix multiplication of user characteristics and book characteristics to get a predictive score (core algorithm)

4.Use MSE (mean square error) algorithm to solve the gradient grads, and use the calculated gradient grads to update the corresponding variables (weights); gradient descent algorithm fits the model
This is implemented in the train_step method：
    @tf.function custom training loop in tensorflow2.0; no need to compile the model, directly use the optimizer to back-propagate the iterative parameters according to the loss function, with the highest flexibility.    

5.'Score prediction' model training and storage:
The checkpoints folder under the models folder --- saves the checkpoints model file for each round of training, which can be iteratively trained on this basis;The folder under the models folder --- Save all the information of the model, you can load it directly without reloading the neural network for prediction

6.The book feature matrix is trained and saved locally:
    6.1 On the basis of the trained mv_net model, obtain the feature map of the input middle layer to obtain the'books layer model books_layer_model'; Use the following method to obtain the feature map of the input middle layer:
      mv_net.model.get_layer("books_combine_layer_flat") --- According to the layer name 'books_combine_layer_flat' Get the layer object;
    6.2 Feed the data into the'books layer model books_layer_model' to get the'book feature vector books_combine_layer_flat_val'
    6.3 Combine the trained'books feature vector books_combine_layer_flat_val' into'books feature matrix books_matrics' and instantiate and save it locally
    6.4 books_matrics.p ---Save the trained book feature matrix to the local

7.The user feature matrix is trained and saved locally:
     Same method as 6;
     users_matrics.p ---The trained user feature matrix is saved locally
     
----------------------------------------------------------------------------------------------------------------

booksRating.pyBook rating prediction:

The model file in the models folder is used for prediction

1.Do matrix multiplication of user characteristics and book characteristics to get predicted scores;

2.Regress the predicted score with the real score, and use MSE to optimize the loss. Because this is essentially a regression problem;

----------------------------------------------------------------------------------------------------------------


modelRecommend.py：

The user feature matrix users_matrics.p and the book feature matrix books_matrics.p are used for prediction

1. Recommend books of the same type
The idea is to calculate the cosine similarity between the feature vector of the current book and the feature matrix of the entire book, and take the top_k with the largest similarity. Here, some random selections are added to ensure that each recommendation is slightly different.。

2. Recommend books you like
The idea is to use the user feature vector and the book feature matrix to calculate the scores of all books, take the top_k with the highest score, and also add some random selection parts.

3. Those who have read this book also read (like) what books
First, select top_k individuals who like a certain book, and obtain the user feature vectors of these individuals.
Then calculate the ratings of these people for all books
Choose the highest-rated book for everyone as a recommendation
Random selection
