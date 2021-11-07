"""Entry point for the HOI_Toolbox. Fir"""
import numpy as np
import pandas as pd
import scipy.io
import time
from toolbox.higher_order_information.Oinfo import OInfoCalculator
from toolbox.higher_order_information.dOinfo import DOInfoCalculator
from toolbox.higher_order_information.local_o_info import LocalOHOI
from toolbox.states_probabilities import StatesProbabilities
from toolbox.utils import save_obj, load_obj


class HOIToolbox:

    def __init__(self, config):
        """Create a new HOI Toolbox object, using a configuration file"""
        self.config = config
        if "metric" not in config:
            print("ERROR : no metric specified")
            exit(1)
        if "input" not in config:
            print("Please provide input data location in config")
            exit(1)
        if "input_type" not in config:
            self.config["input_type"] = "tsv"

    def run(self):
        input_file = "data/" + self.config["input"]
        if self.config["input_type"] == "tsv":
            # df = pd.read_csv("data/timeseries.tsv.gz", compression='gzip', delimiter='\t')
            # df = df.loc[:, (df != 0.0).any(axis=0)]
            # df.to_csv('data/cleaned_timeseries.tsv', sep='\t',index=False)
            ts = np.genfromtxt(input_file, delimiter='\t', )
            self.ts = ts[1:, :].T
        # print(ts.shape)
        elif self.config["input_type"] == "mat":
            ts = scipy.io.loadmat(input_file)
            ts = np.array(ts['ts'])
            self.ts = ts.T
        # print(ts.shape)
        else:
            print("Unknown input type")
            exit(1)
        output_file = self.config["metric"] + "_" + self.config["input"].split(".")[0]

        if self.config["metric"] == "Oinfo":
            t = time.time()
            o_info_calculator = OInfoCalculator(self.config)
            Odict = o_info_calculator.run(self.ts, self.config)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            save_name = self.config["input"].split(".")[0] + "_O"
            print("Saving and trying to load again")
            save_obj(Odict, save_name)
            Odict_Oinfo = load_obj('Odict_Oinfo')
            print("Done.")

        elif self.config["metric"] == "dOinfo":
            t = time.time()
            d_o_info_calculator = DOInfoCalculator(self.config)
            dOdict = d_o_info_calculator.run(self.ts, self.config)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            save_name = self.config["input"].split(".")[0] + "_dO"
            save_obj(dOdict, save_name)
            print("done.")

        elif self.config["metric"] == "local_o":
            t = time.time()
            ts = pd.read_csv("data/"+self.config["input"], sep='\t')
            #ts = pd.DataFrame(self.ts.transpose())
            if "workers" in self.config:
                n_workers = self.config["workers"]
            else:
                n_workers = 8
            local_o_hoi = LocalOHOI(probability_estimator=StatesProbabilities(ts), n_workers=n_workers)
            local_o = local_o_hoi.exhaustive_local_o(ts)
            elapsed = time.time() - t
            print("Elapsed time is ", elapsed, " seconds.")
            print("Saving " + output_file + " and trying to load again")
            save_obj(local_o, output_file)
            local_o = load_obj(output_file)
            print("Done.")
        else:
            print("ERROR : Unknown metric")
            exit(1)
