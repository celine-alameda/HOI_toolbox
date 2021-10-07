import sys
import json
import numpy as np
import itertools
import pandas as pd

from toolbox.utils import load_obj, CombinationsManager

for i in range(1, 160):
    for snd in ["on", "off"]:
        file_name = "scenario1bis_trial_" + str(i) + "_" + snd + ".tsv"
        print("using " + file_name)

        data = pd.read_table("data/" + file_name)
        variable_names = list(data.columns)
        save_name = file_name.split(".")[0]
        local_o_info = load_obj("local_o_"+save_name)
        print("Local o info readout, object {}".format(save_name))

        data["Local o"] = local_o_info["local_o"]
        data["sig"] = local_o_info["significances"]
        data["lower_ci"] = local_o_info["lower_ci"]
        data["upper_ci"] = local_o_info["upper_ci"]
        output_file_name = save_name + ".tsv"
        data.to_csv(output_file_name, sep='\t', index=False)

