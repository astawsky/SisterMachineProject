
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

    print(struct.Sisters[0].keys())

    for ID in range(len(struct.Sisters)):

        # plt.plot(struct.Sisters[ID]['cellareaA']/np.linalg.norm(struct.Sisters[ID]['cellareaA']), label='cellareaA')
        # plt.plot(struct.Sisters[ID]['lengthA']/np.linalg.norm(struct.Sisters[ID]['lengthA']), label='lengthA')
        # plt.plot(struct.Sisters[ID]['fluorescenceA']/np.linalg.norm(struct.Sisters[ID]['fluorescenceA']), label='fluorescenceA')
        plt.plot(struct.Sisters[ID]['fluorescenceA'], label='fluorescenceA')
        plt.plot(struct.Sisters[ID]['fluorescenceB'], label='fluorescenceB')
        plt.legend()
        plt.show()
        # Fluorescence measurements that are lower than ~50,000 are not reliable, the reliable traces all have a bottom ylim of ~50,000


if __name__ == '__main__':
    main()
