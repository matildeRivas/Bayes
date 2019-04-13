import numpy as np
import Tarea1.DataReader as dr
import Tarea1.PlotGenerator as pg


# given the probability of belonging to each class and a
def determine_class(prob, theta):
    if prob >= theta:
        return 0
    else:
        return 1


def class_probability(data, mean, cov, dimensions):
    d = data[:-1]
    mu = d - mean
    return (1 / (np.sqrt(np.power(2 * np.pi, dimensions) * np.linalg.det(cov)))) * (
        np.exp((-1 / 2) * mu.T.dot(np.linalg.inv(cov)).dot(mu)))


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


def gaussian_bayes(training, testing):
    class0, class1 = dr.label_divider(training)
    cov0 = np.cov(class0[:, :-1].T)
    mean0 = np.mean(class0[:, :-1], axis=0)
    cov1 = np.cov(class1[:, :-1].T)
    mean1 = np.mean(class1[:, :-1], axis=0)
    dimensions = len(class1[0]) - 1
    prob_quotient = []
    real_class = testing[:, -1]
    for d in testing:
        class0_prob = class_probability(d, mean0, cov0, dimensions)
        class1_prob = class_probability(d, mean1, cov1, dimensions)
        prob_quotient.append(class0_prob / class1_prob)

    roc = ([0], [0])
    theta = np.logspace(np.log10(max(min(prob_quotient), 10 ** -20)), np.log10(max(prob_quotient)), num=100)
    for t in theta:
        tpr, fpr = count(prob_quotient, real_class, t)
        roc[0].append(tpr)
        roc[1].append(fpr)
    return roc


if __name__ == '__main__':
    training, testing = dr.data_separator(dr.file_reader("magic04_label.data"), 0.8)
    gaussian_bayes(training, testing)
