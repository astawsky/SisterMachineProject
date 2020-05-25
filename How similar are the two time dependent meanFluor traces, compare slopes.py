
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats


# CHECK IF YOU MESSED UP WITH THE X,Y PLACEMENT IN THE ARGUMENTS OF LINREGRESS!!!! IE. RUN IT AGAIN, CHANGED


# For the folder "Trend Analysis of Protein" and with the xlabel and savefig "Trend Analysis of Protein (NEW)"

def GetSlopes(trace, n):

    # Window style, trying to make an algorithm that cuts trace at length that's divisible by n
    remainder = np.mod(len(trace), n)

    if remainder == 0:
        trace = trace
    else:
        trace = trace[:-remainder]

    # stats.linregress gives, in order, slope intercept rvalue pvalue stderr, that's why we put the [0] after
    slopes = np.array([stats.linregress(np.array([trace[ind + l] for l in range(n)]), range(n))[0] for ind in range(len(trace) - n)])
    # This might be a little weird because of discrete timepoints, do we NEED to use the actual time
    # points or can we seperate them via range(len(x))?
    thetas = np.arctan(slopes)

    return slopes, thetas


def CompareSlopes(A_trace, B_trace):

    if len(A_trace) == len(B_trace):
        IOError('A_trace and B_trace are not the same length')

    return PearsonCorr(A_trace,B_trace) # np.sum(A_trace-B_trace)*(1/len(A_trace))


def PearsonCorr(u, v):
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    covariance = np.sum(np.array([(u[ID] - avg_u) * (v[ID] - avg_v) for ID in range(len(u))]))
    denominator = np.sqrt(np.sum(np.array([(u[ID] - avg_u) ** 2 for ID in range(len(u))]))) * \
                  np.sqrt(np.sum(np.array([(v[ID] - avg_v) ** 2 for ID in range(len(v))])))
    weighted_sum = covariance / denominator

    # np.corrcoef(np.array(u), np.array(v))[0, 1] --> Another way to calculate the pcorr coeff using numpy, gives similar answer

    return weighted_sum


def HistOfSlopes(dist_sis, dist_non, dist_both, label_sis, label_non, label_both, Nbins, abs_range, n, Slope):
    arr_sis = plt.hist(x=dist_sis, label=label_sis, weights=np.ones_like(dist_sis) / float(len(dist_sis)), bins=Nbins,
                       range=abs_range)
    arr_non = plt.hist(x=dist_non, label=label_non, weights=np.ones_like(dist_non) / float(len(dist_non)), bins=Nbins,
                       range=abs_range)
    arr_both = plt.hist(x=dist_both, label=label_both, weights=np.ones_like(dist_both) / float(len(dist_both)),
                        bins=Nbins, range=abs_range)
    plt.close()

    # range=[0, abs_diff_range]

    plt.plot(np.array([(arr_sis[1][l] + arr_sis[1][l + 1]) / 2. for l in range(len(arr_sis[1]) - 1)]), arr_sis[0],
             label=label_sis, marker='.')
    plt.plot(np.array([(arr_non[1][l] + arr_non[1][l + 1]) / 2. for l in range(len(arr_non[1]) - 1)]), arr_non[0],
             label=label_non, marker='.')
    plt.plot(np.array([(arr_both[1][l] + arr_both[1][l + 1]) / 2. for l in range(len(arr_both[1]) - 1)]), arr_both[0],
             label=label_both, marker='.')

    if Slope:
        title = 'Slopes Pearson Correlation of slope from lin. reg. of n=' + str(n) + ' points'
    else:
        title = 'Arctan(Slopes) Pearson Correlation of slope from lin. reg. of n=' + str(n) + ' points'

    xlabel = 'Pearson Correlation of slope from lin. reg. of n=' + str(n) + ' points'
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('PDF')
    plt.savefig(title+'.png', dpi=300)
    plt.close()


def main():
    # compare the slopes that we get from the

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    for n in range(2, 18, 1):
        sis_slope_dist = []
        sis_theta_dist = []
        for index in range(len(struct.Sisters)):
            A_slopes, A_thetas = GetSlopes(struct.Sisters[index]['meanfluorescenceA'], n)
            B_slopes, B_thetas = GetSlopes(struct.Sisters[index]['meanfluorescenceB'], n)
            slope_similarity = CompareSlopes(A_slopes, B_slopes)
            theta_similarity = CompareSlopes(A_thetas, B_thetas)
            # print('similarity', similarity)
            # print(index)
            sis_slope_dist.append(slope_similarity)
            sis_theta_dist.append(theta_similarity)

        non_slope_dist = []
        non_theta_dist = []
        for index in range(len(struct.Nonsisters)):
            A_slopes, A_thetas = GetSlopes(struct.Nonsisters[index]['meanfluorescenceA'], n)
            B_slopes, B_thetas = GetSlopes(struct.Nonsisters[index]['meanfluorescenceB'], n)
            slope_similarity = CompareSlopes(A_slopes, B_slopes)
            theta_similarity = CompareSlopes(A_thetas, B_thetas)
            # print('similarity', similarity)
            # print(index)
            non_slope_dist.append(slope_similarity)
            non_theta_dist.append(theta_similarity)

        both_slope_dist = []
        both_theta_dist = []
        for index1, index2 in zip(struct.Control[0], struct.Control[1]):
            minval = min(len(struct.Sisters[index1]['meanfluorescenceA']), len(struct.Nonsisters[index2]['meanfluorescenceA']))
            A_slopes, A_thetas = GetSlopes(struct.Sisters[index1]['meanfluorescenceA'][:minval], n)
            B_slopes, B_thetas = GetSlopes(struct.Nonsisters[index2]['meanfluorescenceA'][:minval], n)
            slope_similarity = CompareSlopes(A_slopes, B_slopes)
            theta_similarity = CompareSlopes(A_thetas, B_thetas)
            # print('similarity', similarity)
            # print(index)
            both_slope_dist.append(slope_similarity)
            both_theta_dist.append(theta_similarity)


        # For the raw slope
        # calculate the complete range to get all the histograms in the same x-range
        min_range = min(min(sis_slope_dist), min(non_slope_dist), min(both_slope_dist))
        max_range = max(max(sis_slope_dist), max(non_slope_dist), max(both_slope_dist))
        abs_range = [min_range, max_range]

        # calculate the means and variance for all the dsets
        sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(sis_slope_dist)) + r', $\sigma=$' + '{:.2e}'.format(
            np.nanstd(sis_slope_dist))
        non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(non_slope_dist)) + r', $\sigma=$' + \
                    '{:.2e}'.format(np.nanstd(non_slope_dist))
        both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(both_slope_dist)) + r', $\sigma=$' + \
                     '{:.2e}'.format(np.nanstd(both_slope_dist))

        # Parameter, number of boxes in histogram, that I found to be the most insightful
        Nbins = 13

        # graphs all together
        HistOfSlopes(sis_slope_dist, non_slope_dist, both_slope_dist, sis_label, non_label, both_label, Nbins, abs_range, n, True)


        # For the theta instead of slope!
        # calculate the complete range to get all the histograms in the same x-range
        min_range = min(min(sis_theta_dist), min(non_theta_dist), min(both_theta_dist))
        max_range = max(max(sis_theta_dist), max(non_theta_dist), max(both_theta_dist))
        abs_range = [min_range, max_range]

        # calculate the means and variance for all the dsets
        sis_label = 'Sister ' + r'$\mu=$' + '{:.2e}'.format(
            np.nanmean(sis_theta_dist)) + r', $\sigma=$' + '{:.2e}'.format(
            np.nanstd(sis_theta_dist))
        non_label = 'Non Sister ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(non_theta_dist)) + r', $\sigma=$' + \
                    '{:.2e}'.format(np.nanstd(non_theta_dist))
        both_label = 'Control ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(both_theta_dist)) + r', $\sigma=$' + \
                     '{:.2e}'.format(np.nanstd(both_theta_dist))

        # Parameter, number of boxes in histogram, that I found to be the most insightful
        Nbins = 13

        # graphs all together
        HistOfSlopes(sis_theta_dist, non_theta_dist, both_theta_dist, sis_label, non_label, both_label, Nbins,
                     abs_range, n, False)


if __name__ == '__main__':
    main()
