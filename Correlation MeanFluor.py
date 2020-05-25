
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.signal as signal


def CrossCorr(in1, in2):

    # They have to be the same length
    if len(in1)!=len(in2):
        IOError('Problem!')

    # Length of array
    L = len(in1)

    # Means of the traces independent of lag
    mean_1 = np.mean(in1)
    mean_2 = np.mean(in2)

    x = in1 - mean_1
    y = in2 - mean_2
    # x = x / np.linalg.norm(x)
    # y = y / np.linalg.norm(y)

    # print(np.linalg.norm(x), np.linalg.norm(y))

    corr = []
    for lag in range(int(np.ceil(3*L/4))):
        if lag == 0:
            l, k = x, y
        else:
            l, k = x.iloc[lag:], y.iloc[:-lag]

        # print(len(l))
        positive_lag = np.dot(l, k)/(np.std(l)*np.std(k)*len(l))

        if lag == 0:
            l, k = x, y
        else:
            l, k = x[:-lag], y[lag:]
        negative_lag = np.dot(l, k) / (np.std(l)*np.std(k)*len(l))

        # positive_lag = (np.sum(in1[lag:]*in2[:-lag])-mean_1*mean_2)/(np.std(in1[lag:]-mean_1)*np.std(in2[:-lag]-mean_2))
        # negative_lag = (np.sum(in1[:-lag] * in2[lag:])-mean_1*mean_2) / (np.std(in1[:-lag]-mean_1) * np.std(in2[lag:]-mean_2))
        # positive_lag = np.sum(in1[lag:]*in2[:-lag])/(np.sqrt(np.sum(in1[lag:]**2)+np.sum(in2[:-lag]**2)))
        # negative_lag = np.sum(in1[:-lag] * in2[lag:]) / (np.sqrt(np.sum(in1[:-lag] ** 2) + np.sum(in2[lag:] ** 2)))
        # negative_lag = (np.sum(in1[:-lag] * in2[lag:]) / (L - lag) - mean_1 * mean_2) / (np.sum(in1 * in2) - mean_1 * mean_2)
        average_lag = (positive_lag+negative_lag)/2
        corr.append(average_lag)

    corr = np.array(corr)

    return corr


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    for index, index1, index2, index3 in zip(range(len(struct.Sisters)), range(len(struct.Nonsisters) - 1), np.flip(struct.Control[0], 0),
                                             np.flip(struct.Control[1], 0)):
        x = struct.Sisters[index]['meanfluorescenceA']
        y = struct.Sisters[index]['meanfluorescenceB']
        min_gen = min(len(x), len(y))
        # Means of the traces independent of lag
        mean_x = np.mean(x[:min_gen])
        mean_y = np.mean(y[:min_gen])

        if len(struct.A_dict_sis['Sister_Trace_A_'+str(index)]) > 20:
            print(len(struct.A_dict_sis['Sister_Trace_A_'+str(index)]))
            # plt.xcorr(x[:min_gen]-mean_x, y[:min_gen]-mean_y, maxlags=int(np.ceil(3*min_gen/4)), normed=True)
            corr = CrossCorr(x[:min_gen], y[:min_gen])
            # corr = signal.correlate(x[:min_gen], y[:min_gen], mode='same')
            plt.plot(corr)
            plt.grid(True)
            plt.ylabel('Normalized Cross Correlation Coefficient of Mean Fluorescence')
            plt.title('Sisters')
            plt.xlabel('Lag in Absolute time')
            plt.xlim([0,int(np.ceil(3*min_gen/4))])
            plt.show()

            plt.plot(np.arange(len(x-mean_x)), x-mean_x)
            plt.plot(np.arange(len(y-mean_y)), y-mean_y)
            plt.show()

        # x = struct.Nonsisters[index1]['meanfluorescenceA']
        # y = struct.Nonsisters[index1]['meanfluorescenceB']
        # min_gen = min(len(x), len(y))
        # # Means of the traces independent of lag
        # mean_x = np.mean(x[:min_gen])
        # mean_y = np.mean(y[:min_gen])
        #
        # # plt.xcorr(x[:min_gen]-mean_x, y[:min_gen]-mean_y, maxlags=int(np.ceil(3*min_gen/4)), normed=True)
        # corr = CrossCorr(x[:min_gen], y[:min_gen])
        # # corr = signal.correlate(x[:min_gen], y[:min_gen], mode='same') / min_gen
        # plt.plot(corr)
        # plt.grid(True)
        # plt.ylabel('Normalized Cross Correlation Coefficient of Mean Fluorescence')
        # plt.title('Non-Sisters')
        # plt.xlabel('Lag in Absolute time')
        # plt.xlim([0, int(np.ceil(3 * min_gen / 4))])
        # plt.show()
        #
        # x = struct.Sisters[index2]['meanfluorescenceA']
        # y = struct.Nonsisters[index3]['meanfluorescenceB']
        # min_gen = min(len(x), len(y))
        # # Means of the traces independent of lag
        # mean_x = np.mean(x[:min_gen])
        # mean_y = np.mean(y[:min_gen])
        #
        # # plt.xcorr(x[:min_gen]-mean_x, y[:min_gen]-mean_y, maxlags=int(np.ceil(3*min_gen/4)), normed=True)
        # corr = CrossCorr(x[:min_gen], y[:min_gen])
        # # corr = signal.correlate(x[:min_gen], y[:min_gen], mode='same') / min_gen
        # plt.plot(corr)
        # plt.grid(True)
        # plt.ylabel('Normalized Cross Correlation Coefficient of Mean Fluorescence')
        # plt.title('Control')
        # plt.xlabel('Lag in Absolute time')
        # plt.xlim([0, int(np.ceil(3 * min_gen / 4))])
        # plt.show()


if __name__ == '__main__':
    main()
