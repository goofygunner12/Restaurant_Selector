# Created: 26-May-2016
# Author: Vishwas K. Mruthyunjaya
# Program: Restaurant Selector
import csv
import sys
import numpy as np
import time
from scipy.spatial.distance import cdist
from sklearn import svm
import random


# main function
def main():
    # sys.stdout.write("Initialising the variables and formatting the data...\n\n")
    sys.stdout.write("Starting the simulation with a random origin...\n\n")
    # Initialise the variables
    ratings_data = []
    restaurant_data = []
    text_file = open("distance_and_ratings.txt", "w")
    # Read the ratings.txt file
    with open(sys.argv[1], 'rb') as csvfile:
        file_reader_1 = csv.reader(csvfile, delimiter=';', quotechar='|')
        for every_row_1 in file_reader_1:
            ratings_data.append([element.strip() for element in every_row_1])

    # Read the restaurant.txt file
    with open(sys.argv[2], 'rb') as csvfile:
        file_reader_2 = csv.reader(csvfile, delimiter=';', quotechar='|')
        for every_row_2 in file_reader_2:
            restaurant_data.append([element.strip() for element in every_row_2])
    counter = 0
    while (counter<1000):
        counter += 1
        # Convert the raw data from text file to numpy array
        ratings_data = np.array(ratings_data)
        restaurant_data = np.array(restaurant_data)
        # get ratings and distance
        my_ratings = get_my_ratings(ratings_data).astype(float)
        my_ratings = np.reshape(my_ratings, (my_ratings.shape[0], 1))
        restaurant_distance, dict_id_index, dict_id_name, origin = get_restaurant_distance(restaurant_data)
        table = np.array(np.concatenate((my_ratings, restaurant_distance), axis=1))
        my_ratings = my_ratings / my_ratings.max(axis=0)
        # get SVM testing data
        svm_test_data = np.concatenate((my_ratings, restaurant_distance), axis=1)
        # Run SVM classifier to select a restaurant
        # sys.stdout.write("Running Restaurant Selector...\n\n")
        restaurant_scores = select_restaurant(svm_test_data)
        # sys.stdout.write("Restaurants have been scored. Selecting a restaurant...\n\n")
        selected_restaurant = dict_id_name[dict_id_index[np.argmax(restaurant_scores)]]
        # sys.stdout.write("RESTAURANT SELECTED: %s" % selected_restaurant)
        text_file.write("%s; %s; %s; %s\n" % (selected_restaurant, origin, my_ratings[np.argmax(restaurant_scores)] * 5.0
                        ,restaurant_distance[np.argmax(restaurant_scores)]))
    sys.stdout.write("END OF SIMULATION.\n")
    text_file.close()



# Function to select the restaurant
def select_restaurant(svm_test_data):
    x = np.array([[0.2, 1], [1, 0], [0.0, 1]])
    y = [0, 1, 0]
    classifier = svm.SVR()
    # classifier = svm.SVC(kernel='precomputed')
    # classifier = svm.SVC(kernel='linear', C = 1.0)
    classifier.fit(x, y)
    restaurant_scores = classifier.predict(svm_test_data)
    return restaurant_scores


# Function to get the distance from the given coordinates
def get_restaurant_distance(restaurant_data):
    dict_id_index = dict(enumerate(restaurant_data[1:][:, 0]))
    dict_id_name = dict(zip((restaurant_data[1:][:, 0]), restaurant_data[1:][:, 1]))
    coordinates = restaurant_data[1:, [2, 3]]
    #origin = np.array([[0.0, 0.0]])
    origin = np.array([[random.uniform(-4.5, 5.5), random.uniform(-4.9, 5.8)]])
    restaurant_distance = cdist(coordinates, origin, 'euclidean')
    restaurant_distance = restaurant_distance / restaurant_distance.max(axis=0)
    return restaurant_distance, dict_id_index, dict_id_name, origin


# Function to get "my" ratings for those restaurants without "my" ratings
def get_my_ratings(ratings_data):
    # A loop to get the data for Neural Network (MLP)
    data_for_neural_nets = []
    # get output matrix for neural nets training
    output_neural_nets = ratings_data[1, ][1:]
    output_neural_nets = np.array(output_neural_nets)
    indices_no_ratings = np.argwhere(output_neural_nets == '')

    # get input matrix for neural network training and testing
    # NOTE: Testing set data is that without any ratings by "ME"
    for i in range(1, ratings_data.shape[1]):
        # Initialise a temp variable
        temp_data = []

        # Get the restaurant ratings by all the reviewers
        restaurant_ratings = ratings_data[2:][:, i]
        restaurant_ratings = restaurant_ratings[restaurant_ratings != '']
        restaurant_ratings = restaurant_ratings.astype(float)

        # Calculate the average, number of reviewers, and standard deviation
        # of the ratings by the reviewers for each restaurant
        temp_data.append((np.average(restaurant_ratings))/5.00)
        temp_data.append((np.std(restaurant_ratings))/3.00)
        temp_data.append(float(len(restaurant_ratings)/100.00))

        # Store the AVG, STD.DEV, and COUNT as parameters to Neural Network
        data_for_neural_nets.append(temp_data)

    # Convert to numpy array
    data_for_neural_nets = np.array(data_for_neural_nets)

    # Initialise training and dev data variables for neural network
    training_input_neural_nets = []
    training_output_neural_nets = []
    dev_data_neural_nets = []

    # Split the data for Neural Network into training and dev set
    for j in range(0, len(data_for_neural_nets)):
        if j not in indices_no_ratings:
            training_input_neural_nets.append((data_for_neural_nets[j][:]))
            training_output_neural_nets.append(float(output_neural_nets[j])/5.00)
        else:
            dev_data_neural_nets.append(data_for_neural_nets[j][:])

    # Convert into numpy array
    training_input_neural_nets = np.array(training_input_neural_nets)
    training_output_neural_nets = np.array(training_output_neural_nets)
    dev_data_neural_nets = np.array(dev_data_neural_nets)

    # Initialise random weights
    # weights_ij_train --> weights connecting between the input and hidden nodes
    # weights_jk_train --> weights connecting between the hidden and output nodes
    num_hidden_units = 3
    learning_rate = 0.08
    training_output_neural_nets = np.reshape(training_output_neural_nets, (training_output_neural_nets.shape[0], 1))
    weights_ij_train = np.random.uniform(-0.05, 0.05, (training_input_neural_nets.shape[1], num_hidden_units))
    weights_jk_train = np.random.uniform(-0.05, 0.05, (num_hidden_units, 1))
    # System Message
    # sys.stdout.write("Running ANN to predict 'MY' ratings for those restaurants 'I' have not visited or rated...\n\n")
    # Function Call
    # Training the neural nets to obtain the weights
    weights_ij_test, weights_jk_test = recursive_back_propagation(learning_rate,
                                                                  training_input_neural_nets,
                                                                  training_output_neural_nets.astype(float),
                                                                  weights_ij_train,
                                                                  weights_jk_train)
    # Predicting the ratings of those restaurants using the network weights from training
    predicted_ratings = network_testing(weights_ij_test,
                                        weights_jk_test,
                                        dev_data_neural_nets)
    output_neural_nets = update_my_ratings(predicted_ratings, indices_no_ratings, output_neural_nets)
    return output_neural_nets


# Recursive Back-Propagation Algorithm
def recursive_back_propagation(lr, xI, t, wIJ_, wJK_):
    wIJ = np.copy(wIJ_)
    wJK = np.copy(wJK_)
    OI = xI
    timeLimit = time.time() + 5
    while time.time() < timeLimit:
        # compute output of each layer (OI, OJ, and OK)
        OJ = sigmoid(np.dot(OI, wIJ))
        OK = sigmoid(np.dot(OJ, wJK))
        # get delta k
        diff_t_OK = t - OK
        OK_der = OK * (1 - OK)
        delta_k = diff_t_OK * OK_der
        # get delta j
        sum_deltaK_wJK = delta_k.dot(np.transpose(wJK))
        OJ_der = OJ * (1 - OJ)
        delta_j = sum_deltaK_wJK * OJ_der
        # update weights
        delta_wIJ = lr * np.transpose(OI).dot(delta_j)
        delta_wJK = lr * np.transpose(OJ).dot(delta_k)
        wIJ += delta_wIJ
        wJK += delta_wJK
        # calculate the error equation summation(t - OK)^2
        # error = t*t + OK*OK - 2*t*OK
        # sys.stdout.write("%s\n" % str(np.mean(np.abs(error))))
    # return trained weights
    return wIJ, wJK


# function to test the network
def network_testing(weights_ij_test, weights_jk_test, test_data):
    # convert the raw data to numpy array
    testing_data = np.array(test_data)
    # Calculate the output
    oi_test = testing_data
    oj_test = sigmoid(np.dot(oi_test, weights_ij_test))
    ok_test = sigmoid(np.dot(oj_test, weights_jk_test))
    return ok_test


def update_my_ratings(predicted_ratings, indices_to_update, output_neural_nets):
    for i in range(0, predicted_ratings.shape[0]):
        output_neural_nets[indices_to_update[i]] = predicted_ratings[i] * 5.0
    return output_neural_nets


# Sigmoid function to calculate sigmoid of a matrix
def sigmoid(matrix):
    # sigma(n) = 1 / (1 + e^-x)
    return 1.0 / (1.0 + np.exp(-1.0 * matrix))

# Calling main()
if __name__ == "__main__":
    main()