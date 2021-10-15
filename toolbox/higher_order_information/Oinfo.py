import numpy as np
import itertools
from tqdm.auto import tqdm
from toolbox.estimator.gcmi_estimator import copnorm
from toolbox.higher_order_information.HOI import HOI
from toolbox.utils import bootci, CombinationsManager, ncr, boot_samples, bootstrap_ci


class OInfoCalculator(HOI):

    def __init__(self, config):
        super().__init__(config)

    def get_ent(self, x):
        entropy = self.estimator.estimate_entropy(x)
        return entropy

    def o_information_boot(self, data, indices_sample, indices_variables):
        # this function takes the whole X as input, and additionally the indices
        # convenient for bootstrap
        # X size is M(variables) x N (samples)

        # print("1 ",X.shape)
        data = data[indices_variables, :]
        data = data[:, indices_sample]

        m = data.shape[0]
        o = (m - 2) * self.get_ent(data)

        for j in range(m):
            x1 = np.delete(data, j, axis=0)
            o = o + self.get_ent(data[j, :]) - self.get_ent(x1)
        return o

    def run(self, ts, config):
        higher_order = config["higher_order"]
        Xfull = copnorm(ts)
        n_variables, n_observations = Xfull.shape
        print("Timeseries details - Number of variables: ", str(n_variables), ", Number of timepoints: ",
              str(n_observations))
        print("Computing Oinfo using " + self.estimator.type + " estimator")
        X = Xfull
        maxsize = config["maxsize"]  # 5 # max number of variables in the multiplet
        n_best = config["n_best"]  # 10 # number of most informative multiplets retained
        nboot = config["nboot"]  # 100 # number of bootstrap samples
        alphaval = 0.05
        o_b = np.zeros((nboot, 1))

        Odict = {}
        # this section is for the expansion of redundancy, so maximizing the O
        # there's no need to fix the target here
        bar_length = (maxsize + 1 - 3)
        with tqdm(total=bar_length) as pbar:
            pbar.set_description("Outer loops, maximizing O")
            for nplet_size in tqdm(range(3, maxsize + 1), disable=True):
                o_for_nplet_size = {}
                if higher_order:
                    combinations_manager = CombinationsManager(n_variables, nplet_size)
                    n_combinations = ncr(n_variables, nplet_size)
                    o_positive = np.zeros(n_best)
                    o_negative = np.zeros(n_best)
                    ind_pos = np.zeros(n_best)
                    ind_neg = np.zeros(n_best)
                else:
                    nplets_iter = itertools.combinations(range(1, n_variables + 1), nplet_size)
                    nplets = []
                    for nplet in nplets_iter:
                        nplets.append(nplet)
                    combinations = np.array(nplets)  # n-tuples without repetition over N modules
                    n_combinations = combinations.shape[0]
                o_array = np.zeros(n_combinations)

                for combination_index in tqdm(range(n_combinations), desc="Inner loop, computing O", leave=False):
                    if higher_order:
                        comb = combinations_manager.nextchoose()
                        o_array = self.o_information_boot(X, range(n_observations), comb - 1)
                        valpos, ipos = np.min(o_positive), np.argmin(o_positive)
                        valneg, ineg = np.max(o_negative), np.argmax(o_negative)
                        if o_array > 0 and o_array > valpos:
                            o_positive[ipos] = o_array
                            ind_pos[ipos] = combinations_manager.combination2number(comb)
                        if o_array < 0 and o_array < valneg:
                            o_negative[ineg] = o_array
                            ind_neg[ineg] = combinations_manager.combination2number(comb)
                    else:
                        comb = combinations[combination_index, :]
                        o_array[combination_index] = self.o_information_boot(X, range(n_observations), comb - 1)

                if higher_order:
                    Osort_pos, ind_pos_sort = np.sort(o_positive)[::-1], np.argsort(o_positive)[::-1]
                    Osort_neg, ind_neg_sort = np.sort(o_negative), np.argsort(o_negative)
                else:
                    ind_pos = np.argwhere(o_array > 0)
                    ind_neg = np.argwhere(o_array < 0)
                    o_positive = o_array[o_array > 0]
                    o_negative = o_array[o_array < 0]
                    Osort_pos, ind_pos_sort = np.sort(o_positive)[::-1], np.argsort(o_positive)[::-1]
                    Osort_neg, ind_neg_sort = np.sort(o_negative), np.argsort(o_negative)

                if Osort_pos.size != 0:
                    n_sel = min(n_best, len(Osort_pos))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        if higher_order:
                            indvar = combinations_manager.number2combination(ind_pos[ind_pos_sort[isel]])
                        else:
                            indvar = np.squeeze(combinations[ind_pos[ind_pos_sort[isel]], :])
                        samples = boot_samples(nboot, range(n_observations))
                        o_values = []
                        for sample in sample:
                            o_value = self.o_information_boot(X, sample, indvar - 1)
                            o_values.append(o_value)
                        ci = bootstrap_ci(o_values, alphaval)
                        ci_lower = ci[0]
                        ci_upper = ci[1]
                        # f = lambda xsamp: self.o_information_boot(X, xsamp, indvar - 1)
                        # ci_lower, ci_upper = bootci(nboot, f, range(n_observations), alphaval)
                        boot_sig[isel] = ci_lower > 0 or ci_upper < 0
                    o_for_nplet_size['sorted_red'] = Osort_pos[0:n_sel]
                    o_for_nplet_size['index_red'] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                    o_for_nplet_size['bootsig_red'] = boot_sig
                if Osort_neg.size != 0:
                    n_sel = min(n_best, len(Osort_neg))
                    boot_sig = np.zeros((n_sel, 1))
                    for isel in range(n_sel):
                        if higher_order:
                            indvar = combinations_manager.number2combination(ind_neg[ind_neg_sort[isel]])
                        else:
                            # All combinations with a negative O, in order.
                            indvar = np.squeeze(combinations[ind_neg[ind_neg_sort[isel]], :])
                        # -1 because combinations start at 1
                        # todo replace this lambda like in lines 114-117
                        f = lambda xsamp: self.o_information_boot(X, xsamp, indvar - 1)
                        ci_lower, ci_upper = bootci(nboot, f, range(n_observations), alphaval)
                        boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                    o_for_nplet_size['sorted_syn'] = Osort_neg[0:n_sel]
                    o_for_nplet_size['index_syn'] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                    o_for_nplet_size['bootsig_syn'] = boot_sig
                Odict[nplet_size] = o_for_nplet_size
                pbar.update(1)

        return Odict
