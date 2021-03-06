import matplotlib.pyplot as plt
import numpy as np


def plot_rocs(rocs):
    lines = []
    for i in rocs:
        roc = i[0]
        tpr, fpr = roc[0], roc[1]
        bucket = i[1]
        label = str(bucket) + " bins"
        line, = plt.plot(fpr, tpr, "-", label=label)
        lines.append(line)

    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas R.O.C. para Naive Bayes")
    plt.legend(handles=lines)
    plt.show()


def plot_roc(roc):
    tpr, fpr = roc[0], roc[1]
    fig = plt.figure(1)
    plt.plot(fpr, tpr, "-")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas R.O.C. para Bayes con Gaussiana")
    plt.show()
    fig.savefig("gauss.png", dpi=1000)

def plot_q(probs):
    x = np.zeros(len(probs))
    plt.plot(x, probs, 'o')
    plt.show()
