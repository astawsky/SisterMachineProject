
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats

# For the folder "Trend Analysis of Protein Correlation by Generation" and "Trend Analysis of Protein Correlation by Generation (NEW)"


def GetSlopesGen(traceA, traceB, n, indices_A, indices_B):

    # number of generations for the analysis we'll compute in this pair
    min_gens = min(len(indices_A), len(indices_B))

    # the points of the continuous signal but split up into generation
    gen_traces_A = [traceA[indices_A[l]:indices_A[l+1]] for l in range(min_gens-1)]
    gen_traces_B = [traceB[indices_B[l]:indices_B[l+1]] for l in range(min_gens-1)]

    # number of points to use inside the generation
    min_points_in_gen = [min(len(gen_traces_A[gen]), len(gen_traces_B[gen])) for gen in range(min_gens-1)]

    # Computing all the angle-regressions that fit into the amount of points in the A/B Trace
    angles_by_gen_A = [np.arctan(np.array([stats.linregress(range(n), np.array([gen_traces_A[gen].iloc[ind + l] for l in range(n)]))[0]
                     for ind in range(min_points_in_gen[gen] - n)])) for gen in range(min_gens-1)]
    angles_by_gen_B = [
        np.arctan(np.array([stats.linregress(range(n), np.array([gen_traces_B[gen].iloc[ind + l] for l in range(n)]))[0]
                            for ind in range(min_points_in_gen[gen] - n)])) for gen in range(min_gens - 1)]

    return angles_by_gen_A, angles_by_gen_B


def PearsonCorr(u, v):

    avg_u = np.mean(u)
    avg_v = np.mean(v)
    covariance = np.sum(np.array([(u[ID] - avg_u) * (v[ID] - avg_v) for ID in range(len(u))]))
    denominator = np.sqrt(np.sum(np.array([(u[ID] - avg_u) ** 2 for ID in range(len(u))]))) * \
                  np.sqrt(np.sum(np.array([(v[ID] - avg_v) ** 2 for ID in range(len(v))])))
    weighted_sum = covariance / denominator

    # np.corrcoef(np.array(u), np.array(v))[0, 1] --> Another way to calculate the pcorr coeff using numpy, gives similar answer

    return weighted_sum


def main():
    # Compare the Pearson Correlations that we get from taking the slopes inside of certain generation of a dataset

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # To plot them all three datasets together since they are calculated seperately
    Control_plot_array = []

    # "n" is the number of points used for the, pi/2 < angle = arctan(slope) < pi/2, regression
    for n in range(2, 11, 1):

        # What we use to compile all the A/B pair slopes in the same generation across all pairs in the dataset
        angles_by_gen_A_array = []
        angles_by_gen_B_array = []
        for indexA, indexB in zip(struct.Control[0], struct.Control[1]):
            # The division times of trace A/B in absolute time
            div_time_A = np.insert(np.array(struct.Sisters[indexA]['timeA'][0] + np.cumsum(np.array(
                struct.A_dict_both['Sister_Trace_A_' + str(indexA)]['generationtime']))), 0,
                                   struct.Sisters[indexA]['timeA'][0])
            div_time_B = np.insert(np.array(struct.Nonsisters[indexB]['timeA'][0] + np.cumsum(np.array(
                struct.B_dict_both['Non-sister_Trace_A_' + str(indexB)]['generationtime']))), 0,
                                   struct.Nonsisters[indexB]['timeA'][0])

            # indices in the dataframe of the division times
            indices_A = np.concatenate([np.where(round(struct.Sisters[indexA]['timeA'], 2) == round(x, 2)) for x in
                                        div_time_A], axis=1)[0]
            indices_B = np.concatenate([np.where(round(struct.Nonsisters[indexB]['timeA'], 2) == round(x, 2)) for x in
                                        div_time_B], axis=1)[0]

            # By Generation, the same size array of fitted angles ready to be compared one to the other
            angles_by_gen_A, angles_by_gen_B = GetSlopesGen(struct.Sisters[indexA]['meanfluorescenceA'],
                                                            struct.Nonsisters[indexB]['meanfluorescenceA'], n, indices_A,
                                                            indices_B)

            angles_by_gen_A_array.append(angles_by_gen_A)
            angles_by_gen_B_array.append(angles_by_gen_B)

        # The list we are going to use to put all the Pearson Correlations that are separated by generation
        angle_similarity_array = []
        for generation in range(50):

            # the angle pairs from all experiments, for a single generation
            angles_A = np.concatenate([angles_by_gen_A_array[index][generation] for index in
                                       range(len(angles_by_gen_A_array)) if
                                       len(angles_by_gen_A_array[index]) > generation])
            angles_B = np.concatenate([angles_by_gen_B_array[index][generation] for index in
                                       range(len(angles_by_gen_B_array)) if
                                       len(angles_by_gen_B_array[index]) > generation])

            # use all of these to get the Pearson Correlation for this generation for all pairs in the data set
            angle_similarity = PearsonCorr(angles_A, angles_B)
            angle_similarity_array.append(angle_similarity)

        # To later plot all datasets together
        Control_plot_array.append(angle_similarity_array)

        # Plotting just this dataset
        # plt.plot(range(20), angle_similarity_array[:20], marker='.')
        # plt.xlabel('Generation number')
        # plt.ylabel('Pearson Correlation of slopes with points n=' + str(n))
        # plt.title('Control')
        # plt.savefig('Control n='+str(n)+' pcorr generations', dpi=300)
        # plt.close()

    # To plot them all three datasets together since they are calculated seperately
    non_plot_array = []

    # "n" is the number of points used for the, pi/2 < angle = arctan(slope) < pi/2, regression
    for n in range(2, 11, 1):

        # What we use to compile all the A/B pair slopes in the same generation across all pairs in the dataset
        angles_by_gen_A_array = []
        angles_by_gen_B_array = []
        for index in range(len(struct.Nonsisters)-1):

            # The division times of trace A/B in absolute time
            div_time_A = np.insert(np.array(struct.Nonsisters[index]['timeA'][0] + np.cumsum(np.array(
                struct.A_dict_non_sis['Non-sister_Trace_A_' + str(index)]['generationtime']))), 0,
                                   struct.Nonsisters[index]['timeA'][0])
            div_time_B = np.insert(np.array(struct.Nonsisters[index]['timeB'][0] + np.cumsum(np.array(
                struct.B_dict_non_sis['Non-sister_Trace_B_' + str(index)]['generationtime']))), 0,
                                   struct.Nonsisters[index]['timeB'][0])

            # indices in the dataframe of the division times
            indices_A = np.concatenate([np.where(round(struct.Nonsisters[index]['timeA'], 2) == round(x, 2)) for x in
                                        div_time_A], axis=1)[0]
            indices_B = np.concatenate([np.where(round(struct.Nonsisters[index]['timeB'], 2) == round(x, 2)) for x in
                                        div_time_B], axis=1)[0]

            # By Generation, the same size array of fitted angles ready to be compared one to the other
            angles_by_gen_A, angles_by_gen_B = GetSlopesGen(struct.Nonsisters[index]['meanfluorescenceA'],
                                                            struct.Nonsisters[index]['meanfluorescenceB'], n,
                                                            indices_A, indices_B)

            # Compile all the A/B pair slopes in the same generation across all pairs in dataset
            angles_by_gen_A_array.append(angles_by_gen_A)
            angles_by_gen_B_array.append(angles_by_gen_B)

        # The list we are going to use to put all the Pearson Correlations that are separated by generation
        angle_similarity_array = []
        for generation in range(50):

            # the angle pairs from all experiments, for a single generation
            angles_A = np.concatenate([angles_by_gen_A_array[index][generation] for index in
                                       range(len(angles_by_gen_A_array)) if
                                       len(angles_by_gen_A_array[index]) > generation])
            angles_B = np.concatenate([angles_by_gen_B_array[index][generation] for index in
                                       range(len(angles_by_gen_B_array)) if
                                       len(angles_by_gen_B_array[index]) > generation])

            # use all of these to get the Pearson Correlation for this generation for all pairs in the data set
            angle_similarity = PearsonCorr(angles_A, angles_B)
            angle_similarity_array.append(angle_similarity)

        # To later plot all datasets together
        non_plot_array.append(angle_similarity_array)

        # Plot just this dataset
        # plt.plot(range(20), angle_similarity_array[:20], marker='.')
        # plt.xlabel('Generation number')
        # plt.ylabel('Pearson Correlation of slopes with points n=' + str(n))
        # plt.title('Non-Sisters')
        # plt.savefig('Non-Sisters n=' + str(n) + ' pcorr generations', dpi=300)
        # plt.close()

    # To plot them all three datasets together since they are calculated seperately
    sis_plot_array = []

    # "n" is the number of points used for the, pi/2 < angle = arctan(slope) < pi/2, regression
    for n in range(2, 11, 1):

        # What we use to compile all the A/B pair slopes in the same generation across all pairs in the dataset
        angles_by_gen_A_array = []
        angles_by_gen_B_array = []
        for index in range(len(struct.Sisters)):

            # The division times of trace A/B in absolute time
            div_time_A = np.insert(np.array(struct.Sisters[index]['timeA'][0] + np.cumsum(np.array(
                struct.A_dict_sis['Sister_Trace_A_' + str(index)]['generationtime']))), 0, struct.Sisters[index]['timeA'][0])
            div_time_B = np.insert(np.array(struct.Sisters[index]['timeB'][0] + np.cumsum(np.array(
                struct.B_dict_sis['Sister_Trace_B_' + str(index)]['generationtime']))), 0,
                                   struct.Sisters[index]['timeB'][0])

            # indices in the dataframe of the division times
            indices_A = np.concatenate([np.where(round(struct.Sisters[index]['timeA'], 2) == round(x, 2)) for x in 
                                      div_time_A], axis=1)[0]
            indices_B = np.concatenate([np.where(round(struct.Sisters[index]['timeB'], 2) == round(x, 2)) for x in
                                        div_time_B], axis=1)[0]

            # By Generation, the same size array of fitted angles ready to be compared one to the other
            angles_by_gen_A, angles_by_gen_B = GetSlopesGen(struct.Sisters[index]['meanfluorescenceA'],
                        struct.Sisters[index]['meanfluorescenceB'], n, indices_A, indices_B)

            # Compile all the A/B pair slopes in the same generation across all pairs in dataset
            angles_by_gen_A_array.append(angles_by_gen_A)
            angles_by_gen_B_array.append(angles_by_gen_B)

        # The list we are going to use to put all the Pearson Correlations that are separated by generation
        angle_similarity_array = []
        for generation in range(50):

            # the angle pairs from all experiments, for a single generation
            angles_A = np.concatenate([angles_by_gen_A_array[index][generation] for index in
                        range(len(angles_by_gen_A_array)) if len(angles_by_gen_A_array[index]) > generation])
            angles_B = np.concatenate([angles_by_gen_B_array[index][generation] for index in
                        range(len(angles_by_gen_B_array)) if len(angles_by_gen_B_array[index]) > generation])

            # use all of these to get the Pearson Correlation for this generation for all pairs in the data set
            angle_similarity = PearsonCorr(angles_A, angles_B)
            angle_similarity_array.append(angle_similarity)

        # To later plot all datasets together
        sis_plot_array.append(angle_similarity_array)

        # Plot just this dataset
        # plt.plot(range(20), angle_similarity_array[:20], marker='.')
        # plt.xlabel('Generation number')
        # plt.ylabel('Pearson Correlation of slopes with points n='+str(n))
        # plt.savefig('Sisters n=' + str(n) + ' pcorr generations', dpi=300)
        # plt.close()

    # Plotting them all together
    for n in range(len(sis_plot_array)):
        plt.plot(range(20), sis_plot_array[n][:20], marker = '.', label='Sister')
        plt.plot(range(20), non_plot_array[n][:20], marker='.', label='Non-Sister')
        plt.plot(range(20), Control_plot_array[n][:20], marker='.', label='Control')
        plt.xticks(range(20))
        plt.ylim([-.3,.3])
        plt.ylabel('Pearson Correlation Value, n='+str(n+2))
        plt.xlabel('Generation Number')
        plt.legend()
        plt.savefig('All together n='+str(n+2)+' pcorr generations', dpi=300)
        plt.close()


if __name__ == '__main__':
    main()
