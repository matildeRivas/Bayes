from Tarea1.NaiveBayes import naive_bayes
from Tarea1.GaussianBayes import gaussian_bayes
import Tarea1.DataReader as dr

import matplotlib.pyplot as plt


def tarea1(training, testing):
    naive_rocs = naive_bayes(training, testing, [410])
    gauss_roc = gaussian_bayes(training, testing)
    lines = []
    for i in naive_rocs:
        roc = i[0]
        tpr, fpr = roc[0], roc[1]
        bucket = i[1]
        label = "con histogramas"
        line, = plt.plot(fpr, tpr, "-", label=label)
        lines.append(line)
    tpr, fpr = gauss_roc[0], gauss_roc[1]
    line, = plt.plot(fpr, tpr, "-", label="con Gaussiana")
    lines.append(line)
    fig = plt.figure(1)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Clasificador Bayesiano")
    plt.legend(handles=lines)
    plt.show()
    fig.savefig("Bayes.png", dpi=1000)

training, testing = dr.data_separator(dr.file_reader("magic04_label.data"), 0.8)
tarea1(training, testing)