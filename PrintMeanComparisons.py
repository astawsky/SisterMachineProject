
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

from scipy.interpolate import spline

def main():


    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    sis_means, sis_stds, sis_dist, non_means, non_stds, non_dist, both_mean, both_stds, both_dist\
        = metadatastruct.CompMeans()

    Nbins = 10

    # HISTOGRAM
    for param, xmax in zip(list(sis_means.keys()), [.25, 1.5, 3, .5, 1.5]):

        sis_label = 'Sister '+r'$\mu=$'+'{:.2e}'.format(np.mean(sis_dist[param]))+r', $\sigma=$'+'{:.2e}'.format(np.var(sis_dist[param]))
        non_label = 'Non Sister '+r'$\mu=$'+'{:.2e}'.format(np.mean(non_dist[param]))+r', $\sigma=$'+'{:.2e}'.format(np.var(non_dist[param]))
        both_label = 'Control '+r'$\mu=$'+'{:.2e}'.format(np.mean(both_dist[param]))+r', $\sigma=$'+'{:.2e}'.format(np.var(both_dist[param]))

        plt.figure()
        plt.hist(sis_dist[param], label=sis_label, range=[0, xmax], bins=Nbins,
                       weights= np.ones_like(sis_dist[param])/float(len(sis_dist[param])))
        plt.title('Distribution of the difference between A and B in mean ' + param)
        plt.xlabel('value of the difference in mean')
        plt.ylabel('number of samples that have this value')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/Sisters/'
                    + str(param), dpi = 300)
        plt.close()
        plt.figure()
        plt.hist(non_dist[param], label=non_label, range=[0, xmax], bins=Nbins,
                       weights= np.ones_like(non_dist[param])/float(len(non_dist[param])))
        plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
        plt.xlabel('value of the difference in mean')
        plt.ylabel('number of samples that have this value')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/NonSisters/'
                    + str(param), dpi = 300)
        plt.close()
        plt.figure()
        plt.hist(both_dist[param], label=both_label, range=[0, xmax], bins=Nbins,
                       weights= np.ones_like(both_dist[param])/float(len(both_dist[param])))
        plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
        plt.xlabel('value of the difference in mean')
        plt.ylabel('number of samples that have this value')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/Random Control/'
                    + str(param), dpi = 300)

        plt.figure()
        if str(param) == 'division_ratios__f_n':
            arr = plt.hist(x=sis_dist[param], label=sis_label, alpha=0.33, range=[0, .2], bins=Nbins,
                           weights=np.ones_like(sis_dist[param]) / float(len(sis_dist[param])))
            arr1 = plt.hist(x=non_dist[param], label=non_label, alpha=0.33, range=[0, .2], bins=Nbins,
                            weights=np.ones_like(non_dist[param]) / float(len(non_dist[param])))
            arr2 = plt.hist(x=both_dist[param], label=both_label, alpha=0.33, range=[0, .2], bins=Nbins,
                            weights=np.ones_like(both_dist[param]) / float(len(both_dist[param])))
            plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
            plt.xlabel('value of the difference in mean')
            plt.ylabel('number of samples that have this value')
            plt.legend()
            plt.savefig(
                '/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/All Together/'
                + str(param), dpi=300)
        else:
            arr = plt.hist(x=sis_dist[param], label=sis_label, alpha=0.33, range=[0, xmax], bins=Nbins,
                           weights= np.ones_like(sis_dist[param])/float(len(sis_dist[param])))
            arr1 = plt.hist(x=non_dist[param], label=non_label, alpha=0.33, range=[0, xmax], bins=Nbins,
                            weights= np.ones_like(non_dist[param])/float(len(non_dist[param])))
            arr2 = plt.hist(x=both_dist[param], label=both_label, alpha=0.33, range=[0, xmax], bins=Nbins,
                            weights= np.ones_like(both_dist[param])/float(len(both_dist[param])))
            plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
            plt.xlabel('value of the difference in mean')
            plt.ylabel('number of samples that have this value')
            plt.legend()
            plt.savefig(
                '/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/All Together/'
                + str(param), dpi = 300)

        xnew = np.linspace(arr[1][1:].min(), arr[1][1:].max(), 300)
        xnew1 = np.linspace(arr1[1][1:].min(), arr1[1][1:].max(), 300)
        xnew2 = np.linspace(arr2[1][1:].min(), arr2[1][1:].max(), 300)

        power_smooth = spline(arr[1][1:], arr[0], xnew)
        power_smooth1 = spline(arr1[1][1:], arr1[0], xnew1)
        power_smooth2 = spline(arr2[1][1:], arr2[0], xnew2)

        plt.figure()
        plt.plot(xnew, power_smooth, label=sis_label)
        plt.plot(xnew1, power_smooth1, label=non_label)
        plt.plot(xnew2, power_smooth2, label=both_label)
        plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
        plt.xlabel('value of the difference in mean')
        plt.ylabel('number of samples that have this value')
        plt.legend()
        plt.savefig(
            '/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/All Together/'
            +'splined '+ str(param), dpi=300)

        plt.figure()
        plt.plot(arr[1][1:], arr[0], label=sis_label, marker='.')
        plt.plot(arr1[1][1:], arr1[0], label=non_label, marker='.')
        plt.plot(arr2[1][1:], arr2[0], label=both_label, marker='.')
        plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
        plt.xlabel('value of the difference in mean')
        plt.ylabel('number of samples that have this value')
        plt.legend()
        plt.savefig(
            '/Users/alestawsky/PycharmProjects/untitled/(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/All Together/'
            + 'straight lines ' + str(param), dpi=300)


        # print('has to be equal to 1', sum(values), sum(values1), sum(values2))
        # print(edges)



if __name__ == '__main__':
    main()
