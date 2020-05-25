
from __future__ import print_function

import matplotlib.pyplot as plt

import pickle

def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (SISTERS)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_sis.keys())[:87], list(metadatastruct.B_dict_sis.keys())[:87]):
        plt.figure()
        l1, l2 = metadatastruct.IndParamNorm(trajA = metadatastruct.A_dict_sis[keyA], trajB = metadatastruct.B_dict_sis[keyB])
        plt.plot(l1, marker='.', label='l1-norm')
        plt.title(keyA + ' and ' + keyB)
        plt.ylim([0,10])
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('Individual Param-Norm Difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared_param_norm/Sisters/'
                    + str(ind), dpi=300)
        plt.close()
        ind=ind+1


    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (Non)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_non_sis.keys())[:87], list(metadatastruct.B_dict_non_sis.keys())[:87]):
        plt.figure()
        l1, l2 = metadatastruct.IndParamNorm(trajA = metadatastruct.A_dict_non_sis[keyA], trajB = metadatastruct.B_dict_non_sis[keyB])
        plt.plot(l1, marker='.', label='l1-norm')
        plt.title(keyA + ' and ' + keyB)
        plt.ylim([0,10])
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('Individual Param-Norm Difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared_param_norm/Non/'
                    + str(ind), dpi=300)
        plt.close()
        ind=ind+1

    
    # PLOTTING THE CYCLE PARAMETERS OF A AND B IN THE SAME GRAPH (Control)
    ind = 0
    for keyA, keyB in zip(list(metadatastruct.A_dict_both.keys())[:87], list(metadatastruct.B_dict_both.keys())[:87]):
        plt.figure()
        l1, l2 = metadatastruct.IndParamNorm(trajA = metadatastruct.A_dict_both[keyA], trajB = metadatastruct.B_dict_both[keyB])
        plt.plot(l1, marker='.', label='l1-norm')
        plt.title(keyA + ' and ' + keyB)
        plt.ylim([0,10])
        plt.xlabel('Generation (Cycle) Number')
        plt.ylabel('Individual Param-Norm Difference')
        plt.legend()
        plt.savefig('/Users/alestawsky/PycharmProjects/untitled/Cycle_data_compared_param_norm/Control/'
                    + str(ind), dpi=300)
        plt.close()
        ind=ind+1


if __name__ == '__main__':
    main()
