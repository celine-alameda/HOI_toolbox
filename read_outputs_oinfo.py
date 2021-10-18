# This python file has sample codes as to how to read the outputs generated for different configs.
# The output dictionary is slightly different whether you are using higher_order = False or True

# necessary imports
import sys
import json
import numpy as np
import itertools
import pandas as pd

from toolbox.utils import load_obj, CombinationsManager

configFilename = "config.json"
argCount = len(sys.argv)
if (argCount > 1):
    configFilename = sys.argv[1]
with open(configFilename, "r") as fd:
    config = json.load(fd)

data = pd.read_table("data/"+config["input"])
variable_names = list(data.columns)
save_name = config["input"].split(".")[0] + "_O"
o_info_dict = load_obj(save_name)

print("Oinfo readout, object {}".format(save_name))
print(o_info_dict)
n_variables = len(variable_names)
output_data = []
for nplet_size in range(3, n_variables + 1):
    # print(o_info_dict[nplet_size])
    nplets_iter = itertools.combinations(range(1, n_variables + 1), nplet_size)
    nplets = []
    for nplet in nplets_iter:
        nplets.append(nplet)
    combinations_array = np.array(nplets)

    if "index_red" in o_info_dict[nplet_size]:
        red_indices = o_info_dict[nplet_size]["index_red"]
        red_values = o_info_dict[nplet_size]["sorted_red"]
        red_significancies = o_info_dict[nplet_size]["bootsig_red"]
        print("Highest redundancies for nplet of size {} :".format(nplet_size))
        for i in range(len(red_indices)):
            data_row = {}
            nplet = combinations_array[red_indices[i]]
            variable_names_in_output = []
            for k in nplet:
                variable_names_in_output.append(variable_names[k - 1])
            for name in variable_names:
                if name in variable_names_in_output:
                    data_row[name] = 1
                else:
                    data_row[name] = 0
            data_row["Size"] = nplet_size
            data_row["Type"] = "Red"
            data_row["Rank"] = i + 1
            data_row["Sig"] = red_significancies[i][0]
            data_row["Value"] = red_values[i]
            output_data.append(data_row)
            print("{} with value {}".format(variable_names_in_output, red_values[i]))
    if "index_syn" in o_info_dict[nplet_size]:
        syn_indices = o_info_dict[nplet_size]["index_syn"]
        syn_values = o_info_dict[nplet_size]["sorted_syn"]
        syn_significancies = o_info_dict[nplet_size]["bootsig_syn"]
        print("Highest synergies for nplet of size {} :".format(nplet_size))
        for i in range(len(syn_indices)):
            data_row = {}
            nplet = combinations_array[syn_indices[i]]
            variable_names_in_output = []
            for k in nplet:
                variable_names_in_output.append(variable_names[k - 1])
            for name in variable_names:
                if name in variable_names_in_output:
                    data_row[name] = 1
                else:
                    data_row[name] = 0
            data_row["Size"] = nplet_size
            data_row["Type"] = "Syn"
            data_row["Rank"] = i + 1
            data_row["Value"] = syn_values[i]
            data_row["Sig"] = syn_significancies[i][0]
            output_data.append(data_row)
            print("{} with value {}".format(variable_names_in_output, syn_values[i]))
output_df = pd.DataFrame(output_data)
output_file_name = save_name + ".xlsx"
output_df.to_excel(output_file_name)
