import csv
import numpy as np


# reads a data file
def file_reader(file_name):
    with open(file_name, newline='') as data_file:
        reader = csv.reader(data_file)
        data = np.array(list(reader)).astype("float")
    return data


# divides data in two arrays according to the label value
def label_divider(data):
    label0 = []
    label1 = []
    for d in data:
        if d[-1] == 0:
            label0.append(d)
        else:
            label1.append(d)
    label0 = np.array(label0)
    label1 = np.array(label1)
    return label0, label1


# returns a training and testing set
def data_separator(data, ratio):
    # shuffle data before extracting
    np.random.shuffle(data)
    # separate mixed data by class
    class0, class1 = label_divider(data)
    # fill sets according to ratio
    training = []
    testing = []
    size0 = len(class0)
    size1 = len(class1)
    for i in range(size0):
        if i <= ratio * size0:
            training.append(class0[i])
        else:
            testing.append(class0[i])
    for i in range(size1):
        if i <= ratio * size1:
            training.append(class1[i])
        else:
            testing.append(class1[i])
    # suffle new arrays
    testing = np.array(testing)
    training = np.array(training)
    np.random.shuffle(testing)
    np.random.shuffle(training)
    return training, testing
