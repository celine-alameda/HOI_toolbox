"""Entry point for the HOI_Toolbox. Fir"""
import numpy as np
import pandas as pd
import scipy.io
import time

from toolbox.Oinfo import exhaustive_loop_zerolag
from toolbox.dOinfo import exhaustive_loop_lagged
from toolbox.local_o_info import exhaustive_local_o
from toolbox.utils import save_obj, load_obj


class HOIToolbox:

    def __init__(self, config):
        """Create a new HOI Toolbox object, using a configuration file"""
        self.config = config
        if "metric" in config:
            self.metric = config["metric"]
        else:
            print("ERROR : no metric specified")
            exit(1)

        if "input" in config:
            self.file_name = config["input"]
        else:
            print("Please provide input data location in config")
            exit(1)

        if "input_type" in config:
            self.input_type = config["input_type"]
        else:
            self.input_type = "tsv"

    def run(self):
        input_file = "data/" + self.file_name
        if self.input_type == "tsv":
            # df = pd.read_csv("data/timeseries.tsv.gz", compression='gzip', delimiter='\t')
            # df = df.loc[:, (df != 0.0).any(axis=0)]
            # df.to_csv('data/cleaned_timeseries.tsv', sep='\t',index=False)
            ts = np.genfromtxt(input_file, delimiter='\t', )
            self.ts = ts[1:, :].T
        # print(ts.shape)
        elif self.input_type == "mat":
            ts = scipy.io.loadmat(input_file)
            ts = np.array(ts['ts'])
            self.ts = ts.T
        # print(ts.shape)
        else:
            print("Unknown input type")
            exit(1)
        output_file = self.metric + "_" + self.file_name.split(".")[0]
        if self.metric == "Oinfo":
            print("WARNING : NOT REFACTORED / OPTIMIZED YET")
            t = time.time()
            Odict = exhaustive_loop_zerolag(self.ts, self.config)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            save_name = self.config["input"].split(".")[0] + "_O"
            print("Saving and trying to load again")
            save_obj(Odict, save_name)
            Odict_Oinfo = load_obj('Odict_Oinfo')
            print("Done.")

        elif self.metric == "dOinfo":
            print("WARNING : NOT REFACTORED / OPTIMIZED YET")
            t = time.time()
            Odict = exhaustive_loop_lagged(self.ts, self.config)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            save_obj(Odict, 'Odict_dOinfo')
            Odict_dOinfo = load_obj('Odict_dOinfo')
            print(Odict_dOinfo)

        elif self.metric == "local_o":
            t = time.time()
            ts = pd.DataFrame(self.ts.transpose())
            local_o = exhaustive_local_o(ts)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            print("Saving " + output_file + " and trying to load again")
            save_obj(local_o, output_file)
            local_o = load_obj(output_file)
            print("Done.")
        else:
            print("ERROR : Unknown metric")
            exit(1)
