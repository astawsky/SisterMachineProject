from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt

import pickle

import scipy.stats as stats

import scipy.optimize as optimize


def polyfit_with_fixed_points(n, x, y, xf, yf):
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]


def GetLabels(array, dset):
    # calculate the means and variance for the array and puts it into a string
    label = dset + ' ' + r'$\mu=$' + '{:.2e}'.format(np.nanmean(array)) + r', $\sigma=$' + '{:.2e}'.format(np.nanstd(array))

    return label


def GetFractionalGeneration(debug, sync_array, ID, Starting_ref, Ending_ref, ref_abs_div_times, focal_abs_div_times, ref_gen_num, focal_gen_num):
    # We use the A trace as reference
    for ref_start, ref_end in zip(Starting_ref, Ending_ref):

        focal_before = np.array([focal_start for focal_start in focal_abs_div_times[ID][:-1] if
                                 np.where(focal_abs_div_times[ID] == focal_start) + np.ones_like(focal_start,
                                                                                                 dtype=np.int64) < len(
                                     focal_abs_div_times[ID]) and (
                                             focal_start < ref_start and focal_abs_div_times[ID][np.where(focal_abs_div_times[ID] == focal_start) +
                                                                                                 np.ones_like(focal_start,
                                                                                                              dtype=np.int64)] <= ref_end and
                                             focal_abs_div_times[ID][np.where(focal_abs_div_times[ID] == focal_start) +
                                                                     np.ones_like(focal_start, dtype=np.int64)] > ref_start)])

        # Those that either are 100% synchronized with the reference interval or that fit strictly inside
        focal_inside = np.array(
            [focal_start for focal_start in focal_abs_div_times[ID][:-1] if (focal_start >= ref_start and focal_abs_div_times[ID][np.where(
                focal_abs_div_times[ID] == focal_start) + np.ones_like(focal_start, dtype=np.int64)] <= ref_end)])

        # Those that start inside the reference interval and end on or outside, There can only be one of these
        focal_after = np.array([focal_start for focal_start in focal_abs_div_times[ID][:-1] if
                                (focal_start >= ref_start and focal_start < ref_end and focal_abs_div_times[ID][
                                    np.where(focal_abs_div_times[ID] == focal_start) + np.ones_like(focal_start, dtype=np.int64)] > ref_end)])

        # Those that are bigger in length of interval than the reference interval and encompasses the ref. int., there can only be one of these
        # and, if so, none of the others
        focal_encompass = np.array(
            [focal_start for focal_start in focal_abs_div_times[ID][:-1] if (focal_start < ref_start and focal_abs_div_times[ID][np.where(
                focal_abs_div_times[ID] == focal_start) + np.ones_like(focal_start, dtype=np.int64)] > ref_end)])

        # Checking if there are only one of these...
        if len(focal_before) > 1:
            print('We have more than one element in focal_before!:', focal_before.shape, focal_before)
        if len(focal_after) > 1:
            print('We have more than one element in focal_after!:', focal_after.shape, focal_after)
        if len(focal_encompass) > 1:
            print('We have more than one element in focal_encompass!:', focal_encompass.shape, focal_encompass)

        # all the starting abs. time division points, in order
        all_related_generations = np.concatenate(np.array([focal_before, focal_inside, focal_after, focal_encompass])).astype(np.int64)

        # the reference generation and all the other generations that are associated to it from the other trace
        gen_num_ref = ref_gen_num[np.where(ref_abs_div_times[ID] == ref_start)]
        gen_num_focal = np.array([focal_gen_num[np.where(focal_abs_div_times[ID] == all_related_generations[ind])] for ind in range(len(
            all_related_generations))]).flatten()

        # Number of total abs. time points in the cycle
        ref_trace_num_of_steps = ref_end - ref_start

        # The number of absolute time steps each type of overlap had inside the reference interval, later it will be divided by the number of 
        # time steps inside the reference interval
        before_steps = np.array(focal_abs_div_times[ID][np.where(focal_abs_div_times[ID] == focal_before) + np.ones_like(focal_before,
                                                                                                                         dtype=np.int64)] - ref_start).flatten()
        after_steps = np.array(ref_end - focal_after).flatten()
        inside_steps = np.array([focal_abs_div_times[ID][np.where(focal_abs_div_times[ID] == focal_inside[ind]) + np.ones_like(focal_inside[ind],
                                                                                                                               dtype=np.int64)] -
                                 focal_inside[ind]
                                 for ind in range(len(focal_inside))]).flatten()

        # Necessary so we don't add an empty list to fractional_generation 
        if len(focal_encompass) == 1:
            encompassing_steps = np.array([ref_end - ref_start]).flatten()
        else:
            encompassing_steps = np.array([])

        # The weighted sum, ie. fractional generation
        fractional_generation = np.dot(np.concatenate([before_steps, after_steps, inside_steps, encompassing_steps]) / ref_trace_num_of_steps,
                                       gen_num_focal)

        # Append this comparison to the array that keeps track of this experiment
        sync_array.append(np.array([gen_num_ref, fractional_generation]))

        # This will show some specs that are useful when debugging, like workspace for MATLAB
        if debug:
            print('---------------------------------------')
            print('ref_start', ref_start)
            print('ref_end', ref_end)
            print('focal_before', focal_before)
            print('focal_inside', focal_inside)
            print('focal_after', focal_after)
            print('focal_encompass', focal_encompass)
            print('everything contatenated', all_related_generations)
            print('gen_num_ref', gen_num_ref)
            print('gen_num_focal', gen_num_focal)
            print('before_steps', before_steps)
            print('after_steps', after_steps)
            print('inside_steps', inside_steps)
            print('encompassing_steps', encompassing_steps)
            print('comparing both generations:', gen_num_ref, fractional_generation)

    return sync_array


def main(debug = False):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()


    ## SISTERS ##

    # start_gen = 5

    # Keeps the reference Generation and the fractional Generation per ID (But not a 1-1 mapping!!!!)
    sis_sync_array = []


    abs_div_time_B = np.array([np.cumsum(np.ceil([struct.B_dict_sis[k]['generationtime'] / .05 for k in
                      struct.B_dict_sis.keys() if k == 'Sister_Trace_B_' + str(ID)]).astype(np.int64)) for ID in range(len(struct.Sisters))])
    abs_div_time_A = np.array([np.cumsum(np.ceil([struct.A_dict_sis[k]['generationtime'] / .05 for k in
                            struct.A_dict_sis.keys() if k == 'Sister_Trace_A_' + str(ID)]).astype(np.int64)) for ID in range(len(struct.Sisters))])

    abs_div_time_A = np.array([np.append(np.array([0]), abs_div_time_A[ID]) for ID in range(len(abs_div_time_A))])
    abs_div_time_B = np.array([np.append(np.array([0]), abs_div_time_B[ID]) for ID in range(len(abs_div_time_B))])

    # The Experiment ID, loop over experiments
    for ID in range(len(struct.Sisters)):

        last_gen = min(len(abs_div_time_A[ID]), len(abs_div_time_B[ID]))
        abs_div_time_A[ID] = abs_div_time_A[ID][:last_gen]
        abs_div_time_B[ID] = abs_div_time_B[ID][:last_gen]

        # Decide how many generations each trace has in this experiment
        generation_number_A = np.array([k for k in range(1, last_gen + 1)])
        generation_number_B = np.array([k for k in range(1, last_gen + 1)])

        # where we will save the trace A reference fractional generation
        sis_sync = []

        if debug:
            print('ID:', ID)
            print('both abs div times', abs_div_time_A[ID], abs_div_time_B[ID])

        Starting_ref = np.array([A_starting for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                            abs_div_time_A[ID][1:last_gen - 1]) if A_ending < abs_div_time_B[ID][-1]])
        Ending_ref = np.array([A_ending for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                        abs_div_time_A[ID][1:last_gen - 1]) if A_ending < abs_div_time_B[ID][-1]])

        sis_sync = GetFractionalGeneration(debug=debug, sync_array=sis_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                            ref_abs_div_times=abs_div_time_A, focal_abs_div_times=abs_div_time_B,
                                            ref_gen_num=generation_number_A, focal_gen_num=generation_number_B)


        # save it
        sis_sync_array.append(sis_sync)

        # Use the same variable name, where we will save the trace B reference fractional generation
        sis_sync = []

        # We use the B trace as reference now
        Starting_ref = np.array([B_starting for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                 abs_div_time_B[ID][1:last_gen - 1]) if B_ending < abs_div_time_A[ID][-1]])
        Ending_ref = np.array([B_ending for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                 abs_div_time_B[ID][1:last_gen - 1]) if B_ending < abs_div_time_A[ID][-1]])


        sis_sync = GetFractionalGeneration(debug=debug, sync_array=sis_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                            ref_abs_div_times=abs_div_time_B, focal_abs_div_times=abs_div_time_A,
                                            ref_gen_num=generation_number_B, focal_gen_num=generation_number_A)


        # save it
        sis_sync_array.append(sis_sync)

    ## NONSISTERS ##

    # start_gen = 5

    # Keeps the reference Generation and the fractional Generation per ID (But not a 1-1 mapping!!!!)
    non_sis_sync_array = []

    abs_div_time_B = np.array([np.cumsum(np.ceil([struct.B_dict_non_sis[k]['generationtime'] / .05 for k in
                                                  struct.B_dict_non_sis.keys() if k == 'Non-sister_Trace_B_' + str(ID)]).astype(np.int64)) for ID in
                               range(len(struct.Nonsisters))])
    abs_div_time_A = np.array([np.cumsum(np.ceil([struct.A_dict_non_sis[k]['generationtime'] / .05 for k in
                                                  struct.A_dict_non_sis.keys() if k == 'Non-sister_Trace_A_' + str(ID)]).astype(np.int64)) for ID in
                               range(len(struct.Nonsisters))])

    abs_div_time_A = np.array([np.append(np.array([0]), abs_div_time_A[ID]) for ID in range(len(abs_div_time_A))])
    abs_div_time_B = np.array([np.append(np.array([0]), abs_div_time_B[ID]) for ID in range(len(abs_div_time_B))])

    # The Experiment ID, loop over experiments
    for ID in range(len(struct.Nonsisters)):

        last_gen = min(len(abs_div_time_A[ID]), len(abs_div_time_B[ID]))
        abs_div_time_A[ID] = abs_div_time_A[ID][:last_gen]
        abs_div_time_B[ID] = abs_div_time_B[ID][:last_gen]

        # Decide how many generations each trace has in this experiment
        generation_number_A = np.array([k for k in range(1, last_gen + 1)])
        generation_number_B = np.array([k for k in range(1, last_gen + 1)])

        # where we will save the trace A reference fractional generation
        non_sis_sync = []

        if debug:
            print('ID:', ID)
            print('both abs div times', abs_div_time_A[ID], abs_div_time_B[ID])

        Starting_ref = np.array([A_starting for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                            abs_div_time_A[ID][1:last_gen - 1]) if
                                 A_ending < abs_div_time_B[ID][-1]])
        Ending_ref = np.array([A_ending for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                        abs_div_time_A[ID][1:last_gen - 1]) if A_ending < abs_div_time_B[ID][-1]])

        non_sis_sync = GetFractionalGeneration(debug=debug, sync_array=non_sis_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                            ref_abs_div_times=abs_div_time_A, focal_abs_div_times=abs_div_time_B,
                                            ref_gen_num=generation_number_A, focal_gen_num=generation_number_B)

        # save it
        non_sis_sync_array.append(non_sis_sync)

        # Use the same variable name, where we will save the trace B reference fractional generation
        non_sis_sync = []

        # We use the B trace as reference now
        Starting_ref = np.array([B_starting for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                                                            abs_div_time_B[ID][1:last_gen - 1]) if
                                 B_ending < abs_div_time_A[ID][-1]])
        Ending_ref = np.array([B_ending for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                                                        abs_div_time_B[ID][1:last_gen - 1]) if B_ending < abs_div_time_A[ID][-1]])

        non_sis_sync = GetFractionalGeneration(debug=debug, sync_array=non_sis_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                            ref_abs_div_times=abs_div_time_B, focal_abs_div_times=abs_div_time_A,
                                            ref_gen_num=generation_number_B, focal_gen_num=generation_number_A)

        # save it
        non_sis_sync_array.append(non_sis_sync)

    ## CONTROL ##

    # start_gen = 5

    # Keeps the reference Generation and the fractional Generation per ID (But not a 1-1 mapping!!!!)
    both_sync_array = []

    abs_div_time_B = np.array([np.cumsum(np.ceil([struct.A_dict_both[k]['generationtime'] / .05 for k in
                                                  struct.A_dict_both.keys() if k == 'Sister_Trace_A_' + str(ID)]).astype(np.int64)) for ID
                               in struct.Control[0]])
    abs_div_time_A = np.array([np.cumsum(np.ceil([struct.B_dict_both[k]['generationtime'] / .05 for k in
                                                  struct.B_dict_both.keys() if k == 'Non-sister_Trace_A_' + str(ID)]).astype(np.int64)) for ID
                               in struct.Control[1]])

    abs_div_time_A = np.array([np.append(np.array([0]), abs_div_time_A[ID]) for ID in range(len(abs_div_time_A))])
    abs_div_time_B = np.array([np.append(np.array([0]), abs_div_time_B[ID]) for ID in range(len(abs_div_time_B))])

    # The Experiment ID, loop over experiments
    for ID in range(len(struct.Control[0])):

        last_gen = min(len(abs_div_time_A[ID]), len(abs_div_time_B[ID]))
        abs_div_time_A[ID] = abs_div_time_A[ID][:last_gen]
        abs_div_time_B[ID] = abs_div_time_B[ID][:last_gen]

        # Decide how many generations each trace has in this experiment
        generation_number_A = np.array([k for k in range(1, last_gen + 1)])
        generation_number_B = np.array([k for k in range(1, last_gen + 1)])

        # where we will save the trace A reference fractional generation
        both_sync = []

        if debug:
            print('ID:', ID)
            print('both abs div times', abs_div_time_A[ID], abs_div_time_B[ID])

        Starting_ref = np.array([A_starting for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                            abs_div_time_A[ID][1:last_gen - 1]) if
                                 A_ending < abs_div_time_B[ID][-1]])
        Ending_ref = np.array([A_ending for A_starting, A_ending in zip(abs_div_time_A[ID][:last_gen - 2],
                                                                        abs_div_time_A[ID][1:last_gen - 1]) if A_ending < abs_div_time_B[ID][-1]])

        both_sync = GetFractionalGeneration(debug=debug, sync_array=both_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                               ref_abs_div_times=abs_div_time_A, focal_abs_div_times=abs_div_time_B,
                                               ref_gen_num=generation_number_A, focal_gen_num=generation_number_B)

        # save it
        both_sync_array.append(both_sync)

        # Use the same variable name, where we will save the trace B reference fractional generation
        both_sync = []

        # We use the B trace as reference now
        Starting_ref = np.array([B_starting for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                                                            abs_div_time_B[ID][1:last_gen - 1]) if
                                 B_ending < abs_div_time_A[ID][-1]])
        Ending_ref = np.array([B_ending for B_starting, B_ending in zip(abs_div_time_B[ID][:last_gen - 2],
                                                                        abs_div_time_B[ID][1:last_gen - 1]) if B_ending < abs_div_time_A[ID][-1]])

        both_sync = GetFractionalGeneration(debug=debug, sync_array=both_sync, ID=ID, Starting_ref=Starting_ref, Ending_ref=Ending_ref,
                                               ref_abs_div_times=abs_div_time_B, focal_abs_div_times=abs_div_time_A,
                                               ref_gen_num=generation_number_B, focal_gen_num=generation_number_A)

        # save it
        both_sync_array.append(both_sync)


    # Plot the difference for each trajectory comparison for each dataset individually
    for ind in range(len(sis_sync_array)):
        exp_progressive_sync = np.array(
            [sis_sync_array[ind][ind2][0] - sis_sync_array[ind][ind2][1] for ind2 in range(len(sis_sync_array[ind]))])
        plt.plot(range(1, len(exp_progressive_sync)+1), exp_progressive_sync, marker='.', label=str(ind))
    plt.grid(True)
    plt.ylim([-10,10])
    plt.xlim([0, 20])
    plt.xlabel('Generation')
    plt.ylabel('Diff. of fractional and reference generation')
    plt.title('Sisters')
    # plt.show()
    plt.savefig('Sisters, Diff. of fractional and reference generation.png', dpi=300)
    plt.close()

    for ind in range(len(non_sis_sync_array)):
        exp_progressive_sync = np.array([non_sis_sync_array[ind][ind2][0]-non_sis_sync_array[ind][ind2][1] for ind2 in range(len(non_sis_sync_array[ind]))])
        plt.plot(range(1, len(exp_progressive_sync)+1), exp_progressive_sync, marker='.', label=str(ind))
    plt.grid(True)
    plt.ylim([-10,10])
    plt.xlim([0, 20])
    plt.xlabel('Generation')
    plt.ylabel('Diff. of fractional and reference generation')
    plt.title('Non-Sisters')
    # plt.show()
    plt.savefig('Non-Sisters, Diff. of fractional and reference generation.png', dpi=300)
    plt.close()

    for ind in range(len(both_sync_array)):
        exp_progressive_sync = np.array([both_sync_array[ind][ind2][0]-both_sync_array[ind][ind2][1] for ind2 in range(len(both_sync_array[ind]))])
        plt.plot(range(1, len(exp_progressive_sync)+1), exp_progressive_sync, marker='.', label=str(ind))
    plt.grid(True)
    plt.ylim([-10,10])
    plt.xlim([0, 20])
    plt.xlabel('Generation')
    plt.ylabel('Diff. of fractional and reference generation')
    plt.title('Control')
    # plt.show()
    plt.savefig('Control, Diff. of fractional and reference generation.png', dpi=300)
    plt.close()

    #
    # # Plot the variance for each trajectory comparison for each dataset individually
    var_per_gen_sis = [np.var([sis_sync_array[exp][gen][0] - sis_sync_array[exp][gen][1] for exp in range(len(sis_sync_array)) if len(sis_sync_array[
                        exp]) > gen]) for gen in range(161)]
    var_per_gen_non_sis = [np.var([non_sis_sync_array[exp][gen][0] - non_sis_sync_array[exp][gen][1] for exp in range(len(non_sis_sync_array)) if
                                   len(non_sis_sync_array[exp]) > gen])for gen in range(161)]
    var_per_gen_both = [np.var([both_sync_array[exp][gen][0] - both_sync_array[exp][gen][1] for exp in range(len(both_sync_array)) if len(
        both_sync_array[exp]) > gen]) for gen in range(161)]
    #
    # print('var_per_gen_sis[:6]', var_per_gen_sis[:6])
    # print('var_per_gen_non_sis[:6]', var_per_gen_non_sis[:6])
    # print('var_per_gen_control[:6]', var_per_gen_both[:6])

    x = np.arange(len(var_per_gen_sis[:40]))
    y = var_per_gen_sis[:40]

    sigma = np.ones_like(x, dtype=np.float64)*.01
    sigma[0] = 1
    #
    # print(sigma)
    #
    # def f(x, *p):
    #     return np.poly1d(p)(x)
    #
    # p1, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0), sigma=sigma)
    # p2, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0))
    #
    # x2 = np.arange(len(var_per_gen_sis[:40]))
    #
    # plt.plot(x2, f(x2, *p1), "r", label=u"fix three points", marker ='^')
    # plt.plot(x2, f(x2, *p2), "b", label=u"no fix")
    #
    # x = np.arange(len(var_per_gen_non_sis[:40]))
    # y = var_per_gen_non_sis[:40]
    #
    # sigma = np.ones_like(x, dtype=np.float64) * .01
    # sigma[0] = 0.001
    #
    # print(sigma)
    #
    # def f(x, y, *p):
    #     return np.polynomial.polynomial.Polynomial.fit(x, y, 2)
    #
    # p1, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0), sigma=sigma)
    # p2, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0))
    #
    # x2 = np.arange(len(var_per_gen_non_sis[:40]))
    #
    # plt.plot(x2, f(x2, *p1), "r", label=u"fix three points", marker='^')
    # plt.plot(x2, f(x2, *p2), "b", label=u"no fix")

    # x = np.arange(len(var_per_gen_both[:40]))
    # y = var_per_gen_both[:40]
    #
    # sigma = np.ones_like(x, dtype=np.float64) * .01
    # sigma[0] = 0.001
    #
    # print(sigma)
    #
    # def f(x, *p):
    #     return np.poly1d(p)(x)
    #
    # p1, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0), sigma=sigma)
    # p2, _ = optimize.curve_fit(f, x, y, (0, 0, 0, 0, 0))
    #
    # x2 = np.arange(len(var_per_gen_both[:40]))
    #
    # plt.plot(x2, f(x2, *p1), "r", label=u"fix three points", marker='^')
    # plt.plot(x2, f(x2, *p2), "b", label=u"no fix")
    #
    # n, d = len(x), 2

    # params = polyfit_with_fixed_points(n, x, y, xf=np.array([0]), yf=var_per_gen_sis[0])
    # poly = np.polynomial.Polynomial(np.random.rand(d + 1))
    # poly = np.polynomial.Polynomial(params)
    # xx = np.linspace(1, len(var_per_gen_sis[:40]))
    # plt.plot(np.arange(len(var_per_gen_sis[:40])), poly(np.arange(len(var_per_gen_sis[:40]))), '-', label='polyfit of sis')

    print(x, y)
    new_series = np.polynomial.polynomial.Polynomial.fit(x, y, 2, w=sigma)
    new_series = new_series.convert().coef
    plt.plot(x, new_series[0] + new_series[1] * x + new_series[2] * (x ** 2), label='polyfit of sis, coeffs: {:.2e}, {:.2e}, {:.2e}'.format(new_series[0],
                                                                                                                                    new_series[1],
                                                                                                                                    new_series[2]))
    # print(var_per_gen_sis[0], new_series[0])

    x = np.arange(len(var_per_gen_non_sis[:40]))
    y = var_per_gen_non_sis[:40]
    # print(x, y)
    new_series = np.polynomial.polynomial.Polynomial.fit(x, y, 2, w=sigma)
    new_series = new_series.convert().coef
    plt.plot(x, new_series[0] + new_series[1] * x + new_series[2] * (x ** 2), label='polyfit of non_sis, coeffs: {:.2e}, {:.2e}, {:.2e}'.format(new_series[0],
                                                                                                                                    new_series[1],
                                                                                                                                    new_series[2]))

    x = np.arange(len(var_per_gen_both[:40]))
    y = var_per_gen_both[:40]
    # print(x, y)
    new_series = np.polynomial.polynomial.Polynomial.fit(x, y, 2, w=sigma)
    new_series = new_series.convert().coef
    plt.plot(x, new_series[0] + new_series[1] * x + new_series[2] * (x ** 2), label='polyfit of both, coeffs: {:.2e}, {:.2e}, {:.2e}'.format(new_series[0],
                                                                                                                                    new_series[1],
                                                                                                                                    new_series[2]))

    plt.plot(var_per_gen_sis[:40], marker='.', label='Sister')
    plt.plot(var_per_gen_non_sis[:40], marker='.', label='Non-Sister')
    plt.plot(var_per_gen_both[:40], marker='.', label='Control')
    # print(var_per_gen_non_sis[0], new_series[0])

    plt.grid(True)
    plt.xlabel('Generation')
    plt.legend()
    # plt.xlim([0, 40])
    # plt.ylim([0,25])
    plt.ylabel('Variance of Diff. of fractional and reference generation')
    plt.title('All dsets together')
    # plt.show()
    plt.savefig('All together, Variance of Diff. of fractional and reference generation.png', dpi=300)
    plt.close()

    # plt.plot(np.log(range(len(var_per_gen_sis[:40]))), np.log(var_per_gen_sis[:40]), marker='.', label='Sister')
    # plt.plot(np.log(range(len(var_per_gen_non_sis[:40]))), np.log(var_per_gen_non_sis[:40]), marker='.', label='Non-Sister')
    # plt.plot(np.log(range(len(var_per_gen_both[:40]))), np.log(var_per_gen_both[:40]), marker='.', label='Control')
    # plt.grid(True)
    # plt.xlabel('log of Generation')
    # plt.ylabel('log of Variance of Diff. of fractional and reference generation')
    # plt.title('All dsets together')
    # plt.show()


    

    # for ind in range(len(both_synch_array)):
    #     exp_progressive_sync = np.array([both_synch_array[ind][ind2][0]-both_synch_array[ind][ind2][1] for ind2 in range(len(both_synch_array[ind]))])
    #     # print(exp_progressive_sync)
    #     plt.plot(exp_progressive_sync, marker='.')
    # plt.xlabel('Generation')
    # plt.ylabel('Difference between fractional generation and reference generation')
    # plt.title('Control')
    # plt.show()





    """"

        # # Format the cycle parameters like in Lee's paper, and in the POWERPOINT
        # Sister

        gen_time_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_sis[keyB]['generationtime'].loc[start_gen:max_gen]) for keyA, keyB
             in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var(
            [np.sum(struct.A_dict_sis[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_sis[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
             zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
             min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(struct.A_dict_sis[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_sis[keyA]['growth_length'].loc[start_gen:max_gen] -
                                   struct.B_dict_sis[keyB]['generationtime'].loc[start_gen:max_gen] * struct.B_dict_sis[keyB]['growth_length'].loc[start_gen:max_gen])
                            for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                            min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
            struct.B_dict_sis[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys())
                             if min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(np.log(struct.A_dict_sis[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
            struct.B_dict_sis[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
                                zip(struct.A_dict_sis.keys(), struct.B_dict_sis.keys()) if
                                min(len(struct.A_dict_sis[keyA]['generationtime']), len(struct.B_dict_sis[keyB]['generationtime'])) > max_gen])

        sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        sis_diff_array_var.append(np.array(sis_diff_array))

        # Non-Sister
        gen_time_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_non_sis[keyB]['generationtime'].loc[
                                                                          :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_non_sis[keyB]['growth_length'].loc[
                                                                         :max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_non_sis[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_non_sis[keyA]['growth_length'].loc[
                                                                          :max_gen] - struct.B_dict_non_sis[keyB][
                                                                                          'generationtime'].loc[start_gen:max_gen] *
            struct.B_dict_non_sis[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
                struct.B_dict_non_sis[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_non_sis[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_non_sis[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_non_sis.keys(), struct.B_dict_non_sis.keys()) if
            min(len(struct.A_dict_non_sis[keyA]['generationtime']), len(struct.B_dict_non_sis[keyB]['generationtime'])) > max_gen])

        non_sis_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        non_sis_diff_array_var.append(np.array(non_sis_diff_array))

        # Control
        gen_time_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[start_gen:max_gen] - struct.B_dict_both[keyB]['generationtime'].loc[
                                                                       :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        alpha_array = np.var([np.sum(
            struct.A_dict_both[keyA]['growth_length'].loc[start_gen:max_gen] - struct.B_dict_both[keyB]['growth_length'].loc[
                                                                      :max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        phi_array = np.var([np.sum(
            struct.A_dict_both[keyA]['generationtime'].loc[start_gen:max_gen] * struct.A_dict_both[keyA]['growth_length'].loc[
                                                                       :max_gen] - struct.B_dict_both[keyB][
                                                                                       'generationtime'].loc[start_gen:max_gen] *
            struct.B_dict_both[keyB]['growth_length'].loc[start_gen:max_gen]) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        div_ratios = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['division_ratios__f_n'].loc[start_gen:max_gen]) - np.log(
                struct.B_dict_both[keyB]['division_ratios__f_n'].loc[start_gen:max_gen])) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])
        birth_lengths = np.var([np.sum(
            np.log(struct.A_dict_both[keyA]['length_birth'].loc[start_gen:max_gen] / struct.x_avg) - np.log(
                struct.B_dict_both[keyB]['length_birth'].loc[start_gen:max_gen] / struct.x_avg)) for keyA, keyB in
            zip(struct.A_dict_both.keys(), struct.B_dict_both.keys()) if
            min(len(struct.A_dict_both[keyA]['generationtime']), len(struct.B_dict_both[keyB]['generationtime'])) > max_gen])

        both_diff_array = [gen_time_array, alpha_array, phi_array, div_ratios, birth_lengths]
        both_diff_array_var.append(np.array(both_diff_array))

    # Name of the cycle parameters for labels
    param_array = [r'Generation time ($T$)', r'Growth rate ($\alpha$)', r'Accumulation Exponent ($\phi$)', r'Log of Division Ratio ($\log(f)$)',
                   r'Normalized Birth Size ($\log(\frac{x}{x^*})$)']

    # separates up to and only options for data
    filename = ['Environmental Influence, random walk diffusion of '+str(param_array[ind]) for ind in range(len(param_array))]
    title = 'With no Epigenetic influence, starting from the fifth generation'

    for ind in range(len(param_array)):
        # Create the labels for each data set
        label_sis = "Sister" # GetLabels(sis_diff_array_var[ind], "Sister")
        label_non = "Non-Sister" # GetLabels(non_sis_diff_array_var[ind], "Non-Sister")
        label_both = "Control" # GetLabels(both_diff_array_var[ind], "Control")

        # Create the x-axis label
        xlabel = 'generation number'
        ylabel = 'Acc. Var. of diff. in '+str(param_array[ind])

        # Graph the Histograms and save them
        HistOfVarPerGen(np.array([sis_diff_array_var[dex][ind] for dex in range(len(sis_diff_array_var))]),
                        np.array([non_sis_diff_array_var[dex][ind] for dex in range(len(non_sis_diff_array_var))]),
                        np.array([both_diff_array_var[dex][ind] for dex in range(len(both_diff_array_var))]),
                        label_sis, label_non, label_both, xlabel, filename[ind], title, ylabel)

    """

if __name__ == '__main__':
    main(debug = False)
