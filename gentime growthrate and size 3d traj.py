import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D


def main():
    # Import the objects we want
    pickle_in = open("NewSisterCellClass_Population.pickle", "rb")
    Population = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Sister.pickle", "rb")
    Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Nonsister.pickle", "rb")
    Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Control.pickle", "rb")
    Control = pickle.load(pickle_in)
    pickle_in.close()

    S_lb = []
    NS_lb = []
    C_lb = []
    S_gt = []
    NS_gt = []
    C_gt = []
    S_gr = []
    NS_gr = []
    C_gr = []

    sns.set_style("white")
    for A_s, B_s, A_ns, B_ns, A_c, B_c in zip(Sister.A_dict.keys(), Sister.B_dict.keys(), Nonsister.A_dict.keys(), Nonsister.B_dict.keys(),
                                              Control.A_dict.keys(), Control.B_dict.keys()):

        min_S = min(len(Sister.A_dict[A_s]), len(Sister.B_dict[B_s]))
        min_NS = min(len(Nonsister.A_dict[A_ns]), len(Nonsister.B_dict[B_ns]))
        min_C = min(len(Control.A_dict[A_c]), len(Control.B_dict[B_c]))

        S_lb.append(Sister.A_dict[A_s]['length_birth'].iloc[:min_S] - Sister.B_dict[B_s]['length_birth'].iloc[:min_S])
        NS_lb.append(Nonsister.A_dict[A_ns]['length_birth'].iloc[:min_NS] - Nonsister.B_dict[B_ns]['length_birth'].iloc[:min_NS])
        C_lb.append(Control.A_dict[A_c]['length_birth'].iloc[:min_C] - Control.B_dict[B_c]['length_birth'].iloc[:min_C])

        S_gt.append(Sister.A_dict[A_s]['generationtime'].iloc[:min_S] - Sister.B_dict[B_s]['generationtime'].iloc[:min_S])
        NS_gt.append(Nonsister.A_dict[A_ns]['generationtime'].iloc[:min_NS] - Nonsister.B_dict[B_ns]['generationtime'].iloc[:min_NS])
        C_gt.append(Control.A_dict[A_c]['generationtime'].iloc[:min_C] - Control.B_dict[B_c]['generationtime'].iloc[:min_C])

        S_gr.append(Sister.A_dict[A_s]['growth_rate'].iloc[:min_S] - Sister.B_dict[B_s]['growth_rate'].iloc[:min_S])
        NS_gr.append(Nonsister.A_dict[A_ns]['growth_rate'].iloc[:min_NS] - Nonsister.B_dict[B_ns]['growth_rate'].iloc[:min_NS])
        C_gr.append(Control.A_dict[A_c]['growth_rate'].iloc[:min_C] - Control.B_dict[B_c]['growth_rate'].iloc[:min_C])
        # ax = plt.gca(projection="3d")
        # ax.plot(Sister.A_dict[A_s]['length_birth'], Sister.A_dict[A_s]['generationtime'], Sister.A_dict[A_s]['growth_rate'], marker='.',
        #          color='blue')
        # ax.plot(Sister.B_dict[B_s]['length_birth'], Sister.B_dict[B_s]['generationtime'], Sister.B_dict[B_s]['growth_rate'], marker='.',
        #          color='orange')
        # plt.title('Sister, key: '+str(A_s))
        # plt.show()
        # plt.close()


    S_lb = np.array(S_lb)
    print(S_lb[0], len(S_lb.shape))


        # for A_x_s,A_y_s,A_z_s,B_x_s,B_y_s,B_z_s,A_x_ns,A_y_ns,A_z_ns,B_x_ns,B_y_ns,B_z_ns,A_x_c,A_y_c,A_z_c,B_x_c,B_y_c,B_z_c in
        #     zip(Sister.A_dict[A_s])


if __name__ == '__main__':
    main()