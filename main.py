# Author: Pranav Mahajan, 2021

import json
import sys
import os
import pandas as pd
from toolbox.hoi_toolbox import HOIToolbox
from toolbox.utils import save_obj, load_obj
from toolbox.Oinfo import exhaustive_loop_zerolag
from toolbox.dOinfo import exhaustive_loop_lagged
from toolbox.local_o_info import exhaustive_local_o

configFilename = "config.json"
argCount = len(sys.argv)
if (argCount > 1):
    configFilename = sys.argv[1]

outputDirectory = "output"
if (not os.path.exists(outputDirectory)):
    os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
    config = json.load(fd)

tb = HOIToolbox(config)
print(tb.file_name)
for i in range(1, 160):
    file_name = "scenario3_trial_" + str(i) + ".tsv"
    tb.file_name = file_name
    tb.run()

    data = pd.read_table("data/" + file_name)
    variable_names = list(data.columns)
    save_name = "local_o_scenario3_trial_" + str(i)
    local_o_info = load_obj(save_name)

    print("Local o info readout, object {}".format(save_name))
    print(local_o_info)

    data["Local o"] = local_o_info
    output_file_name = save_name + ".xlsx"
    data.to_excel(output_file_name)
