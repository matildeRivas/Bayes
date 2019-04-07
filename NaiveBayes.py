import numpy as np
from sys import argv

import Tarea1.DataReader as dr
import Tarea1.PlotGenerator as pg
from Tarea1.Histogram import Histogram

NUM_BUCKETS = 10


# returns all histograms
def get_all_histograms(data, buckets):
    num_hist = len(data[0]) - 1
    class0, class1 = dr.label_divider(data)
    histograms0 = []
    for i in range(num_hist):
        histograms0.append(Histogram(class0[:, i], buckets))
    histograms1 = []
    for i in range(num_hist):
        histograms1.append(Histogram(class1[:, i], buckets))
    return histograms0, histograms1


def attribute_probability(value, histogram):
    bucket = histogram.get_bucket(value)
    return histogram.get_histogram()[bucket]


# returns probability of belonging to a class, given a data array and the class histograms
def class_probability(data, histograms):
    p = 1
    for i in range(len(data) - 1):
        multiplier = max(0.0000001, attribute_probability(data[i], histograms[i]))
        p = p * multiplier
    return p


# given the probability of belonging to each class and a
def determine_class(prob, theta):
    if prob >= theta:
        return 0
    else:
        return 1


def count(probabilities, real_classes, theta):
    predicted_class = []
    for i in range(len(probabilities)):
        predicted_class.append(determine_class(probabilities[i], theta))
    tp = fp = fn = tn = 0
    for p in range(len(probabilities)):
        if predicted_class[p] == 1 and real_classes[p] == 1:
            tp += 1
        elif predicted_class[p] == 0 and real_classes[p] == 1:
            fn += 1
        elif predicted_class[p] == 1 and real_classes[p] == 0:
            fp += 1
        else:
            tn += 1
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr


def main():
    training, testing = dr.data_separator(dr.file_reader("magic04_label.data"), 0.8)
    roc_array = []
    for b in range(2, 20, 7):
        histograms = get_all_histograms(training, b)
        prob_quotient = []
        real_class = testing[:, -1]
        for d in testing:
            class0_prob = class_probability(d, histograms[0])
            class1_prob = class_probability(d, histograms[1])
            prob_quotient.append(class0_prob / class1_prob)
        roc = ([], [])
        theta = np.logspace(np.log10(min(prob_quotient)), np.log10(max(prob_quotient)))
        for t in theta:
            tpr, fpr = count(prob_quotient, real_class, t)
            roc[0].append(tpr)
            roc[1].append(fpr)
        roc_array.append((roc, b))
    pg.plot_roc(roc_array)


if __name__ == '__main__':
    main()
