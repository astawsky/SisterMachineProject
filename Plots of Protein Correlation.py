
# NOT USED BECAUSE THE DISPLACEMENT DOESN'T MAKE SENSE

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pickle

def Output_Graph(where_to_save=None, dataframeA=None, dataframeB=None, is_it_control=False, title=None):
    # First we have to align them with respect to absolute time if they are not control...
    if not is_it_control:
        # We round because of floating point problems when using np.where()
        timeA = round(dataframeA['timeA'], 2)
        timeB = round(dataframeB['timeB'], 2)
        start_time = max(min(timeA), min(timeB))
        end_time = min(max(timeB), max(timeA))
        A_start_ind = np.where(timeA == start_time)[0][0]
        B_start_ind = np.where(timeB == start_time)[0][0]
        A_end_ind = np.where(timeA == end_time)[0][0]
        B_end_ind = np.where(timeB == end_time)[0][0]

        # Plot the proteins and save the plots, as well as the plot of the cross-correlation
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind], label='trace A', marker='.')
        ax1.plot(dataframeB['meanfluorescenceB'][B_start_ind:B_end_ind], label='trace B', marker='.')
        ax1.legend()
        ax2.axhline(y=0, color='black')
        ax2.plot(CrossCorr(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind],
                           dataframeB['meanfluorescenceB'][B_start_ind:B_end_ind], normalize=True, maxlen=None,
                           enforcelen=False),
                 label='Cross-Correlation', marker='.')
        ax2.legend()
        fig.suptitle(title)
        # plt.savefig(where_to_save, dpi=300)
        plt.show()
        plt.close()

    # If it is Control, since they are not in the same time or place, we don't have to worry about time adjustment
    else:
        A_start_ind = 0
        B_start_ind = 0
        end_time = min(len(dataframeA['timeA'])-1, len(dataframeB['timeA'])-1)
        A_end_ind = end_time
        B_end_ind = end_time

        # Plot the proteins and save the plots, as well as the plot of the cross-correlation
        # Here we use the meanfluroescenceA for both dataframes because that's the way the Control
        # dataset is constructed
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind], label='trace A', marker='.')
        ax1.plot(dataframeB['meanfluorescenceA'][B_start_ind:B_end_ind], label='trace B', marker='.')
        ax1.legend()
        ax2.axhline(y=0, color='black')
        ax2.plot(CrossCorr(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind],
                           dataframeB['meanfluorescenceA'][B_start_ind:B_end_ind], normalize=True, maxlen=None,
                           enforcelen=False),
                 label='Cross-Correlation', marker='.')
        ax2.legend()
        fig.suptitle(title)
        # plt.savefig(where_to_save, dpi=300)
        plt.show()
        plt.close()



    # cross_corr = tools.ccf(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind],
    #           dataframeB['meanfluorescenceB'][B_start_ind:B_end_ind], unbiased=False)
    # ax2.plot(cross_corr, label='cross-correlation between A and B', marker='.')
    # ax2.axvline(x=len(cross_corr) / 2, color='black')

    # ax2.xcorr(dataframeA['meanfluorescenceA'][A_start_ind:A_end_ind],
    #           dataframeB['meanfluorescenceB'][B_start_ind:B_end_ind], label='cross-correlation between A and B',
    #           maxlags=None, usevlines=True)
    # ax2.set_xlim(0,500)

def AutoCorr(u, v, normalize=False, maxlen=None, enforcelen=False):
    # Calculates the Cross Correlation
    ml = np.nanmin([len(v), len(u)])
    xp = u[:ml] - np.mean(v[:ml])
    f = np.fft.fft(xp)
    p = np.array([np.real(d) ** 2 + np.imag(d) ** 2 for d in f])
    pi = np.fft.ifft(p)
    Crosscorr = np.real(pi)[:int(ml / 2)]
    if normalize: Crosscorr *= 1. / np.sum(xp ** 2)
    if enforcelen and not maxlen is None:
        if len(Crosscorr) < maxlen:
            Crosscorr = np.concatenate([Crosscorr, np.zeros(maxlen - len(Crosscorr))])

    return Crosscorr

def CrossCorr(u, v, normalize=True, maxlen=None, enforcelen=False):
    avg_u = np.mean(u)
    avg_v = np.mean(v)
    numerator = np.sum(np.array([(u[ID]-avg_u)*(v[ID]-avg_v) for ID in range(len(u))]))
    denominator = np.sqrt(np.sum(np.array([(u[ID]-avg_u)**2 for ID in range(len(u))]))) *\
                  np.sqrt(np.sum(np.array([(v[ID]-avg_v)**2 for ID in range(len(v))])))
    weighted_sum = numerator/denominator

    return weighted_sum

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Put the sister and non sister raw data into lists
    sis_array = [struct.Sisters[data_id] for data_id in range(len(struct.Sisters))]
    non_array = [struct.Nonsisters[data_id] for data_id in range(len(struct.Nonsisters))]

    # Put the Control raw data into lists
    control_array_A = [struct.Sisters[ids] for ids in struct.Control[0]]
    control_array_B = [struct.Nonsisters[ids] for ids in struct.Control[1]]

    # use index for keeping track when saving the graphs
    # Output graphs of Control
    index = 0
    for controlA, controlB in zip(control_array_A, control_array_B):
        Output_Graph(where_to_save='Control Protein Cross-Correlation ' + str(index), dataframeA=controlA,
                     dataframeB=controlB, is_it_control=True, title='Control')
        index = index + 1

    # use index for keeping track when saving the graphs
    # Output graphs of Sisters
    index = 0
    for sis in sis_array:
        Output_Graph(where_to_save='Sister Protein Cross-Correlation ' + str(index), dataframeA=sis, dataframeB=sis,
                     is_it_control=False, title='Sisters')
        index=index+1

    # use index for keeping track when saving the graphs
    # Output graphs of Non_sisters
    index = 0
    for non in non_array:
        Output_Graph(where_to_save='Non-Sister Protein Cross-Correlation ' + str(index), dataframeA=non, dataframeB=non,
                     is_it_control=False, title='Non-Sisters')
        index = index + 1


        


if __name__ == '__main__':
    main()
