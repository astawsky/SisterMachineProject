
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    for index, index1, index2, index3 in zip(range(len(struct.Sisters)), range(len(struct.Nonsisters)-1), struct.Control[0], struct.Control[1]):
        plt.plot(struct.Sisters[index]['meanfluorescenceA'], label='A', marker='.')
        plt.plot(struct.Sisters[index]['meanfluorescenceB'], label='B', marker='.')
        plt.ylabel('mean fluorescence')
        plt.title('Sisters')
        plt.xlabel('time index')
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(struct.Nonsisters[index1]['meanfluorescenceA'], label='A', marker='.')
        plt.plot(struct.Nonsisters[index1]['meanfluorescenceB'], label='B', marker='.')
        plt.ylabel('mean fluorescence')
        plt.title('Non-Sisters')
        plt.xlabel('time index')
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(struct.Sisters[index2]['meanfluorescenceA'], label='A', marker='.')
        plt.plot(struct.Nonsisters[index3]['meanfluorescenceB'], label='B', marker='.')
        plt.ylabel('mean fluorescence')
        plt.title('Control')
        plt.xlabel('time index')
        plt.legend()
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
