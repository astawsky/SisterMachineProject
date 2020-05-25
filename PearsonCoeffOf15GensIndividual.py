

# # # Not sure how helpful/useful this is...


from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle


def main():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    metadatastruct = pickle.load(pickle_in)
    pickle_in.close()

    # # make the folders to put in the graphs
    # os.mkdir('individual pearson correlation coefficients distributions')
    # for param in metadatastruct.A_dict_sis[list(metadatastruct.A_dict_sis.keys())[0]].keys():
    #     os.mkdir('individual pearson correlation coefficients distributions/'+str(param))


    for num_of_gens in range(5, 50, 5):
        # param_array = np.array([])
        for param in metadatastruct.A_dict_sis[list(metadatastruct.A_dict_sis.keys())[0]].keys():
            param_array = np.array([])
            # s_sample = np.array([])
            # n_sample = np.array([])
            # b_sample = np.array([])
            for dsetA, dsetB in zip(
                    [metadatastruct.A_dict_sis, metadatastruct.A_dict_non_sis, metadatastruct.A_dict_both],
                    [metadatastruct.B_dict_sis, metadatastruct.B_dict_non_sis, metadatastruct.B_dict_both]):
                # print('A: ', dsetA.keys())
                # print('B: ', dsetB.keys())
                howmuch = 0
                for idA, idB in zip(dsetA.keys(), dsetB.keys()):
                    if len(dsetA[idA][param])>num_of_gens and len(dsetB[idB][param])>num_of_gens:
                        # mult_expect = (1/num_of_gens)*sum([dsetA[idA][param][j]*dsetB[idB][param][j] for j in range(num_of_gens)])
                        # A_expect = (1/num_of_gens)*sum([dsetA[idA][param][j] for j in range(num_of_gens)])
                        A_mean = np.mean([dsetA[idA][param][:num_of_gens]])
                        B_mean = np.mean([dsetB[idB][param][:num_of_gens]])
                        pear_corr = np.sum([(dsetA[idA][param][ind]-A_mean)*(dsetB[idB][param][ind]-B_mean) for ind in
                                            range(num_of_gens)])/(np.sqrt(np.sum([(dsetA[idA][param][ind]-A_mean)**2 for ind in
                                            range(num_of_gens)]))*np.sqrt(np.sum([(dsetB[idB][param][ind]-B_mean)**2 for ind in
                                            range(num_of_gens)])))
                        param_array = np.append(param_array, pear_corr)
                        howmuch=howmuch+1
                arr = plt.hist(x=param_array, weights = np.ones_like(param_array)/float(len(param_array)))
                plt.close()
                if dsetA == metadatastruct.A_dict_sis:
                    s_sample = arr[0]
                    s_x = [np.mean([arr[1][k], arr[1][k+1]]) for k in range(len(arr[1])-1)]
                    print('sis',howmuch)
                elif dsetA == metadatastruct.A_dict_non_sis:
                    n_sample = arr[0]
                    n_x = [np.mean([arr[1][k], arr[1][k + 1]]) for k in range(len(arr[1]) - 1)]
                    print('non', howmuch)
                elif dsetA == metadatastruct.A_dict_both:
                    b_sample = arr[0]
                    b_x = [np.mean([arr[1][k], arr[1][k + 1]]) for k in range(len(arr[1]) - 1)]
                    print('both', howmuch)
            plt.plot(s_x, s_sample, marker='.', color='r', label=r'Sister ' + str(param) + r', $\mu=$' + '{:.2e}'.format(np.mean(s_x))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(s_x)))
            plt.ylim([0,.4])
            plt.plot(n_x, n_sample, marker='.', color='g', label=r'Non-Sister ' + str(param) + r', $\mu=$' + '{:.2e}'.format(np.mean(n_x))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(n_x)))
            plt.ylim([0, .4])
            plt.plot(b_x, b_sample, marker='.', color='b', label=r'Control ' + str(param) + r', $\mu=$' + '{:.2e}'.format(np.mean(b_x))
                                  + r', $\sigma=$' + '{:.2e}'.format(np.std(b_x)))
            plt.ylim([0, .4])
            plt.xlabel('value of pearson corr')
            plt.ylabel('PDF ('+str(num_of_gens)+' generations)')
            plt.title('normalized histogram of '+str(param))
            plt.legend()
            # plt.show()
            plt.savefig('individual pearson correlation coefficients distributions/'+str(param)+'/ number of gens: '
                        + str(num_of_gens), dpi=300)
            plt.close()




if __name__ == '__main__':
    main()
