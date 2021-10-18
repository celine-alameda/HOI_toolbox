# This python file has sample codes as to how to read the outputs generated for different configs.
# The output dictionary is slightly different whether you are using higher_order = False or True

# necessary imports
import numpy as np
import itertools

from toolbox.utils import load_obj

## NOTE: If you face any issues with loading pickled outputs then use pickle5 instead of pickle in utils.py


## Case 1: argument higher_order = False. 
# Pregenerating all possible combinations, not using the combinatorial numbering system.
# To follow along,
# Run the toy_example.py on fmri timeseries of just first 10 variables for the following config
# "higher_order": false,
# "estimator": "gcmi",
# "modelorder":3,
# "maxsize":4,
# "n_best":10, 
# "nboot":100

# Run for both Oinfo and dOinfo and you will have generated outputs Odict_Oinfo.pkl and Odict_dOinfo.pkl
# Load the dicts (equivalent of structs from MATLAB)

Odict_dOinfo = load_obj('Odict_dOinfo')
print(Odict_dOinfo)

##################
print("dOinfo readout example, higher_order = False")
# How to read the dOinfo output?
# dOinfo computation additionally requires fixing a target
# So lets say you fix the first variable (ROI/timeseries) as target which is 0 index and
# Say you want to know the outputs for multiplet size 3 
# then you would print the following
# target_var_index = 0
# isize = 2
# index is [0, n-1], but what is written is in [1, n] so conversion needed
target_var_index = 1
isize = 2
print(Odict_dOinfo[target_var_index][isize])

# Which will give the result:
# {'sorted_red': array([0.01387719, 0.00827556, 0.00461848, 0.00423925, 0.00421622,
#        0.004088  , 0.00388924, 0.00359825, 0.0033909 , 0.00250356]), 'index_red': array([22,  6, 33, 32, 23, 18, 25, 28,  9, 30], dtype=int64), 'bootsig_red': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'sorted_syn': array([-0.02292608, -0.01824932, -0.01787444, -0.01531304, -0.01455274,
#        -0.01077729, -0.00781092, -0.00726193, -0.00587415, -0.00477357]), 'index_syn': array([31, 17,  4, 29,  2, 14, 24,  1, 15, 34], dtype=int64), 'bootsig_syn': array([[0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.],
#        [0.]]), 'var_arr': array([ 2,  3,  4,  5,  6,  7,  8,  9, 10])}

# You would notice this has an additional output 'var_arr', rest is exactly the same as Oinfo
# 'var_arr' is the array of variables (or ROIs) created after removing the fixed target variable
# The n-plet combinations are drawn from this var_arr array, this will aid us in retrieving the combination

# In order to get back the combination of the max Redundancy value (of 0.01387719)
# which lies in the index 22, do the following
nvartot = 4
var_arr = Odict_dOinfo[target_var_index][isize]['var_arr']
nplets_iter = itertools.combinations(var_arr, isize)
nplets = []
for nplet in nplets_iter:
    nplets.append(nplet)
C = np.array(nplets) # n-tuples without repetition over N modules


print("highest syn : ")
for i in [1]:
    print(C[i])
print("highest red :")
for i in [2, 0]:
    print(C[i])

print("\n \n")