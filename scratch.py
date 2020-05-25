
from __future__ import print_function
import numpy as np
import argparse
import sys,math
import glob
import matplotlib.pyplot as plt
import pickle
import sistercellclass as ssc
import CALCULATETHEBETAS
import os
import scipy.stats as stats
from scipy.interpolate import spline

# Import the Refined Data
pickle_in = open("metastructdata.pickle", "rb")
self = pickle.load(pickle_in)
pickle_in.close()

param = 'division_ratios__f_n'
Nbins = 10

jd = np.concatenate([np.abs(np.log(self.A_dict_sis[ID1][param]) - np.log(self.B_dict_sis[ID2][param])) for ID1, ID2 in zip(self.A_dict_sis.keys(), self.B_dict_sis.keys())])

jd1 = np.concatenate([np.abs(np.log(self.A_dict_non_sis[ID1][param]) - np.log(self.B_dict_non_sis[ID2][param])) for ID1, ID2 in zip(self.A_dict_non_sis.keys(), self.B_dict_non_sis.keys())])

jd2 = np.concatenate([np.abs(np.log(self.A_dict_both[ID1][param]) - np.log(self.B_dict_both[ID2][param])) for ID1, ID2 in zip(self.A_dict_both.keys(), self.B_dict_both.keys())])

print(len(jd))

sis_label = 'Sister '
non_label = 'Non-Sister '
both_label = 'Control '

arr = plt.hist(x=jd, label=sis_label, alpha=0.33, range=[0, .2], bins=Nbins,
                           weights=np.ones_like(jd) / float(len(jd)))
arr1 = plt.hist(x=jd1, label=non_label, alpha=0.33, range=[0, .2], bins=Nbins,
                weights=np.ones_like(jd1) / float(len(jd1)))
arr2 = plt.hist(x=jd2, label=both_label, alpha=0.33, range=[0, .2], bins=Nbins,
                weights=np.ones_like(jd2) / float(len(jd2)))
plt.close()

plt.figure()
plt.plot(arr[1][1:], arr[0], label=sis_label, marker='.')
plt.plot(arr1[1][1:], arr1[0], label=non_label, marker='.')
plt.plot(arr2[1][1:], arr2[0], label=both_label, marker='.')
plt.title('Dist. of the abs. diff. between A and B in mean ' + param)
plt.xlabel('value of the difference in mean')
plt.ylabel('number of samples that have this value')
plt.legend()
plt.savefig('(Hist1) Sister and NonSister Distribution of (mean(A)-mean(B)) (10 bins) new/|log(fnA)-log(fnB)| all together'
            , dpi =300)
