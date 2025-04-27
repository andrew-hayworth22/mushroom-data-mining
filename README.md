# Mushroom Data Mining

Using [this dataset](https://www.kaggle.com/datasets/uciml/mushroom-classification) from Kaggle, we developed a machine learning model for predicting the edibility of mushrooms based on their physical properties. With this model, we also built a user interface that allows users to input data to guess mushroom edibility.

## Data Preprocessing

internal/data_processing.py includes the logic to one-hot encode the dataset attributes to numerical values.

To run our preprocessing, run the command:

``` make preprocess ```

After this is run, the mushrooms_processed.csv file contains the encoded, pruned dataset

## Data Exploration

internal/data_exploration.py includes our data analysis logic. We trained three machine learning models and evaluated their accuracy:

- Logistic regression model
- Random forest classifier
- SVM classifier

In the end, we found that the random forest classifer provided the highest level of accuracy (100%).

To run our exploration code, run this command:

``` make explore ```

## Data Dimensionality Reduction

internal/data_pruning.py has our logic to limit the dimensionality of our dataset. To properly identify candidates for pruning, we analyzed which features had the most impact on our random forest classifier model. Using a minimum threshold, we were then able to prune attributes that had little to n o impact on our model.

To run our pruning logic run this command:

``` make prune ```

## Generating the Model

To avoid running the code step-by-step and simply get the binary for our model, you can run this command:

``` make generate ```

This preprocess the dataset, prunes unecessary attributes, and generates the binary for our random forest classifier in one command.

## Running the Server

Now that the model is generated, we can run the server that provides a user-friendly interface that allows user's to predict the edibility of mushrooms. To run the server, use this command:

``` make server ```

This allows you to input physical data of mushrooms and run it against the model that was generated