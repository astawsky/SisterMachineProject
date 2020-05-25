
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

    # At least two, basically how continuous do we want to make growth length?
    how_many_points_for_reg = 2

    # Make a growth length that's a continuous variable
    for dset, cycle_data_A, cycle_data_B in zip([struct.Sisters, struct.Nonsisters, struct.Control], [struct.A_dict_sis, struct.A_dict_non_sis,
                                                 struct.A_dict_sis], [struct.B_dict_sis, struct.B_dict_non_sis, struct.B_dict_non_sis]):
            # zip(struct.Sisters, struct.A_dict_sis, struct.Control),
            #                                     zip(struct.A_dict_sis, struct.A_dict_non_sis, struct.A_dict_sis),
            #                                     zip(struct.B_dict_sis, struct.B_dict_non_sis, struct.B_dict_non_sis)):
        print('resetting dset_trace')
        dset_trace = []

        print(len(dset))
        print(dset)

        if dset == struct.Sister:
            print('sister is dataset')
        elif dset == struct.Sister:
            print('non sister is dataset')
        elif dset == struct.Control:
            print('Control is dataset')


        if dset != struct.Control:
            for id in range(len(dset)):#len(dset)
                # plt.plot(np.log(dset[id]['lengthA'][:10]), marker='.')
                # plt.show()

                if cycle_data_A == struct.A_dict_sis and cycle_data_B == struct.B_dict_sis:
                    print('sis_cycle_data')
                    # Figuring out where are the division events
                    division_times_A = np.cumsum(np.append(dset[id]['timeA'][0], cycle_data_A['Sister_Trace_A_'+str(id)]['generationtime'][:-1]))
                    division_times_B = np.cumsum(np.append(dset[id]['timeB'][0], cycle_data_B['Sister_Trace_B_'+str(id)]['generationtime'][:-1]))
                elif cycle_data_A == struct.A_dict_non_sis and cycle_data_B == struct.B_dict_non_sis:
                    # Figuring out where are the division events
                    print('non_sis_cycle_data')
                    division_times_A = np.cumsum(np.append(dset[id]['timeA'][0], cycle_data_A['Non-sister_Trace_A_' + str(id)]['generationtime'][
                                                                                 :-1]))
                    division_times_B = np.cumsum(np.append(dset[id]['timeB'][0], cycle_data_B['Non-sister_Trace_B_' + str(id)]['generationtime'][
                                                                                 :-1]))

                # # Check the division times
                # plt.scatter(division_times_A, cycle_data_A['Sister_Trace_A_'+str(id)]['length_birth'])
                # plt.plot(dset[id]['timeA'], dset[id]['lengthA'])
                # plt.show()

                # Now make sure the points used for growth rate don't go through a division events
                A_ctn_growth_rate = np.array([])
                for start, end in zip(np.arange(0, len(dset[id]['lengthA']) - (len(dset[id]['lengthA']) % how_many_points_for_reg),
                  how_many_points_for_reg), np.arange(how_many_points_for_reg-1, (len(dset[id]['lengthA']) - how_many_points_for_reg) -  (len(dset[id]
                  ['lengthA']) % how_many_points_for_reg), how_many_points_for_reg)):
                    
                    # # if it passes through any one of the division times...
                    # if np.any([dset[id]['timeA'][start]<x and x<=dset[id]['timeA'][end] for x in division_times_A]):
                    #     print('TRESPASSING!')
                    #     print(dset[id]['timeA'][start], dset[id]['timeA'][end])
                        # print(division_times_A)
                        # print('first condition',dset[id]['timeA'][start]<division_times_A)
                        # print('second condition',division_times_A<=dset[id]['timeA'][end])
                        # print('Passes through division time '+str(np.where(np.logical_and(dset[id]['timeA'][start]<division_times_A,
                        #                                                              division_times_A<=dset[id]['timeA'][end])))+' '
                        #       'where start is '+str(dset[id]['timeA'][start])+' and end is '+str(dset[id]['timeA'][end]))
                        #
                        # if len(np.where(np.logical_and(dset[id]['timeA'][start]<division_times_A, division_times_A<=dset[id]['timeA'][end]))) != 1:
                        #     print("PROBLEM! NOT OF LENGTH 1")
                        #     print(np.where(np.logical_and(dset[id]['timeA'][start]<division_times_A, division_times_A<=dset[id]['timeA'][end])))
                        # end = np.where((dset[id]['timeA'][start]<division_times_A) & (division_times_A<=dset[id]['timeA'][end]))[0][0]
                        # print('new end:', end)
                        #
                        # # add the growth rate to the continuous growth rate array
                        # A_ctn_growth_rate = np.append(stats.linregress(np.arange(1, how_many_points_for_reg+1, 1), np.log(dset[id]['lengthA'][
                        #                                                                                         start:end]))[0], A_ctn_growth_rate)
                    if not np.any([dset[id]['timeA'][start] < x and x <= dset[id]['timeA'][end] for x in division_times_A]):
                        # print('no trespassing')
                        # print(dset[id]['timeA'][start], dset[id]['timeA'][end])
                        # print('rounded numbers', round(dset[id]['timeA'][start], 2), round(dset[id]['timeA'][end], 2))
                        # print('', np.linspace(round(dset[id]['timeA'][start], 2), round(dset[id]['timeA'][end], 2), num=how_many_points_for_reg))
                        # print('np.log(dset[id][lengthA][start:end+1]):', np.log(dset[id]['lengthA'][start:end+1]))
                        # add the growth rate to the continuous growth rate array

                        A_ctn_growth_rate = np.append(stats.linregress(np.linspace(round(dset[id]['timeA'][start], 2), round(dset[id]['timeA'][end], 2), num=how_many_points_for_reg),
                                                                       np.log(dset[id]['lengthA'][start:end+1]))[0], A_ctn_growth_rate)


                print('B trace has now started')
                # Now make sure the points used for growth rate don't go through a division events
                B_ctn_growth_rate = np.array([])
                for start, end in zip(np.arange(0, len(dset[id]['lengthB']) - (len(dset[id]['lengthB']) % how_many_points_for_reg),
                                                how_many_points_for_reg), np.arange(how_many_points_for_reg - 1,
                                                                                    (len(dset[id]['lengthB']) - how_many_points_for_reg) - (
                                                                                            len(dset[id]
                                                                                                ['lengthB']) % how_many_points_for_reg),
                                                                                    how_many_points_for_reg)):

                    # if it passes through any one of the division times...
                    # if np.any([dset[id]['timeB'][start] < x and x <= dset[id]['timeB'][end] for x in division_times_B]):
                        # print('TRESPASSING!')
                        # print(dset[id]['timeB'][start], dset[id]['timeB'][end])
                        # print(division_times_B)
                        # print('first condition',dset[id]['timeB'][start]<division_times_B)
                        # print('second condition',division_times_B<=dset[id]['timeB'][end])
                        # print('Passes through division time '+str(np.where(np.logical_and(dset[id]['timeB'][start]<division_times_B,
                        #                                                              division_times_B<=dset[id]['timeB'][end])))+' '
                        #       'where start is '+str(dset[id]['timeB'][start])+' and end is '+str(dset[id]['timeB'][end]))
                        #
                        # if len(np.where(np.logical_and(dset[id]['timeB'][start]<division_times_B, division_times_B<=dset[id]['timeB'][end]))) != 1:
                        #     print("PROBLEM! NOT OF LENGTH 1")
                        #     print(np.where(np.logical_and(dset[id]['timeB'][start]<division_times_B, division_times_B<=dset[id]['timeB'][end])))
                        # end = np.where((dset[id]['timeB'][start]<division_times_B) & (division_times_B<=dset[id]['timeB'][end]))[0][0]
                        # print('new end:', end)
                        #
                        # # add the growth rate to the continuous growth rate array
                        # B_ctn_growth_rate = np.append(stats.linregress(np.arange(1, how_many_points_for_reg+1, 1), np.log(dset[id]['lengthB'][
                        #                                                                                         start:end]))[0], B_ctn_growth_rate)
                    if not np.any([dset[id]['timeB'][start] < x and x <= dset[id]['timeB'][end] for x in division_times_B]):
                        # print('no trespassing')
                        # print(dset[id]['timeB'][start], dset[id]['timeB'][end])
                        # print('rounded numbers', round(dset[id]['timeB'][start], 2), round(dset[id]['timeB'][end], 2))
                        # print('',
                        #       np.linspace(round(dset[id]['timeB'][start], 2), round(dset[id]['timeB'][end], 2), num=how_many_points_for_reg))
                        # print('np.log(dset[id][lengthB][start:end+1]):', np.log(dset[id]['lengthB'][start:end + 1]))
                        # add the growth rate to the continuous growth rate array
                        B_ctn_growth_rate = np.append(stats.linregress(
                            np.linspace(round(dset[id]['timeB'][start], 2), round(dset[id]['timeB'][end], 2), num=how_many_points_for_reg),
                            np.log(dset[id]['lengthB'][start:end + 1]))[0], B_ctn_growth_rate)

                # A_ctn_growth_rate = np.array([ stats.linregress(np.arange(1, how_many_points_for_reg+1, 1), np.log(dset[id]['lengthA'][start:end]))[0]
                # for start, \
                #   end in zip(np.arange(0, len(dset[id]['lengthA']) - (len(dset[id]['lengthA']) % how_many_points_for_reg), how_many_points_for_reg),
                #   np.arange(how_many_points_for_reg, (len(dset[id]['lengthA']) - how_many_points_for_reg) - (len(dset[id]['lengthA']) %
                #   how_many_points_for_reg), how_many_points_for_reg)) ])
                # B_ctn_growth_rate = np.array([stats.linregress(np.arange(1, how_many_points_for_reg + 1, 1), np.log(dset[id]['lengthB'][
                #                                                                                                     start:end]))[0]
                # for
                #                      start, \
                #                                                                                                                                 end in
                #                      zip(np.arange(0, len(dset[id]['lengthB']) - (len(dset[id]['lengthB']) % how_many_points_for_reg),
                #                                    how_many_points_for_reg),
                #                          np.arange(how_many_points_for_reg,
                #                                    (len(dset[id]['lengthB']) - how_many_points_for_reg) - (len(dset[id]['lengthB']) %
                #                                                                                            how_many_points_for_reg),
                #                                    how_many_points_for_reg))])

                # print(A_ctn_growth_rate.shape, B_ctn_growth_rate.shape)
                min_len = min(len(A_ctn_growth_rate), len(B_ctn_growth_rate))
                # plt.plot(A_ctn_growth_rate[:min_len] - B_ctn_growth_rate[:min_len], marker = '.')
                # plt.show()
                dset_trace.append(A_ctn_growth_rate[:min_len] - B_ctn_growth_rate[:min_len])
            dset_trace = np.array(dset_trace)

        variance_per_local_point = [np.var([trace[ind] for trace in dset_trace if len(trace) > ind]) for ind in range(max([len(tracess) for tracess
                                                                                                                           in dset_trace]))]
        plt.plot(variance_per_local_point, marker='.')
        plt.xlabel('time index of local growth rate, approx. 1/2 of absolute time, ie. 0.10 hours')
        plt.ylabel('variance of difference of local growth rate over all pairs')
        plt.show()


        # else:



if __name__ == '__main__':
    main()