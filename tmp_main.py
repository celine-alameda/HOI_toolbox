# Author: Pranav Mahajan, 2021

import json
import sys
import os
import pandas as pd
from toolbox.hoi_toolbox import HOIToolbox
from toolbox.utils import load_obj

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



for i in range(1, 160):
    for cnd in ("pre", "post"):
        file_name = "scenario1bis_trial_" + str(i) + "_" + cnd + ".tsv"
        print("using " + file_name)
        tb.file_name = file_name
        tb.run()

        # data = pd.read_table("data/" + file_name)
        # variable_names = list(data.columns)
        # save_name = file_name.split(".")[0]
        # local_o_info = load_obj("local_o_"+save_name)
        # print("Local o info readout, object {}".format(save_name))

        # output_file_name = save_name + ".xlsx"
        # data.to_excel(output_file_name)
