
# # for dataid in range(len(Sisters)):
# #     # plot both traces
# dataid=32
# # plt.plot(Sisters[dataid]['timeA'], Sisters[dataid]['lengthA'], color='b', linewidth=4)
# plt.plot(Sisters[dataid]['timeB'], Sisters[dataid]['lengthB'], color='r', linewidth=4)
# # div_time_arrayA = Sisters[dataid]['timeA'][0] + np.cumsum(np.array([dict_with_all_sister_traces[k]['generationtime'] for
# #                                     k in dict_with_all_sister_traces.keys() if 'Sister_Trace_A_' + str(dataid) in k]))
# div_time_arrayB = Sisters[dataid]['timeB'][0] + np.cumsum(np.array([dict_with_all_sister_traces[k]['generationtime'] for
#                                     k in dict_with_all_sister_traces.keys() if 'Sister_Trace_B_' + str(dataid) in k]))
# plot the vertical lines
# for div_timeA, div_timeB in zip(div_time_arrayA, div_time_arrayB):
#     # plt.axvline(x=div_timeA, color='b', linewidth=2)
#     plt.axvline(x=div_timeB, color='r', linewidth=2)






# blah = Sisters.CellDivisionTrajectory(9, meth, discretize_by = 'length', sisterdata = None, additional_columns = [])
#
# plt.plot(data[9]['lengthA'])
# plt.plot(data[9]['lengthB'])
# plt.show()

# auto_corr_arrayA = []
# auto_corr_arrayB = []
# for dataID in range(0,11,1):
#     acfA, acfB = data.AutocorrelationFFT(dataID, normalize=False, maxlen=20, enforcelen=False)
#     auto_corr_arrayA.append(acfA)
#     auto_corr_arrayB.append(acfB)
# print(auto_corr_arrayA)
# print(auto_corr_arrayB)





# SistersTrajA, SistersTrajB = Sisters.CellDivisionTrajectory(dataID, meth=meth, discretize_by='length', sisterdata=None,
    #                                               additional_columns=[])
    # row_countA = SistersTrajA.shape[0]
    # row_countB = SistersTrajB.shape[0]
    # row_countA_array.append(row_countA)
    # row_countB_array.append(row_countB)

# print('A mean of', np.mean(row_countA_array), 'with a standard dev. of', np.std(row_countA_array))
# print('A mean of', np.mean(row_countB_array), 'with a standard dev. of', np.std(row_countB_array))



# row_countA_array = []
# row_countB_array = []



# NonsistersTrajA, NonsistersTrajB = Nonsisters.CellDivisionTrajectory(dataID, meth=meth, discretize_by='length', sisterdata=None,
    #                                               additional_columns=[])
    # row_countA = NonsistersTrajA.shape[0]
    # row_countB = NonsistersTrajB.shape[0]
    # row_countA_array.append(row_countA)
    # row_countB_array.append(row_countB)





# print('A mean of', np.mean(row_countA_array), 'with a standard dev. of', np.std(row_countA_array))
# print('A mean of', np.mean(row_countB_array), 'with a standard dev. of', np.std(row_countB_array))





# PLOTS THE BETAS OR SCATTERS THE INDIVIDUAL POINTS IN A DICTIONARY:
# dict_with_all_sister_traces, dict_with_sister_traces_A, dict_with_sister_traces_B,
# dict_with_all_non_sister_traces, dict_with_non_sister_traces_A, dict_with_non_sister_traces_B,
# x_avg = xstar1(Sisters, discretize_by='length', traj='A', both=False)