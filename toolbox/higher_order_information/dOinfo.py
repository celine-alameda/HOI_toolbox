import numpy as np
import itertools
from tqdm.auto import tqdm
from toolbox.estimator.gcmi_estimator import copnorm
from toolbox.higher_order_information.HOI import HOI
from toolbox.utils import bootci, CombinationsManager, ncr


class DOInfoCalculator(HOI):

    def __init__(self, config):
        super().__init__(config)
        self.chunk_length = config["chunk_length"]  # round(n_timepoints / 5) can play around with this
        self.order = config["modelorder"]

    def get_cmi(self, A, B, C):
        cmi = self.estimator.estimate_cmi(A, B, C)
        return cmi

    def compute_dynamic_o(self, Y, X, indstart, indvar):
        return self.o_information_lagged_boot(Y, X, indstart, 0, indvar)

    def o_information_lagged_boot(self, Y, X, indstart, chunklength, indvar):
        # evaluates the o_information flow
        # Y Nx1 target vector .  X NxM drivers
        # m order of the model
        ### chunklength is used only for bootstrapping, otherwise it is 0
        if chunklength == 0:
            indsample = np.arange(len(Y))
        else:
            nchunks = int(np.floor(len(Y) / chunklength))
            indstart = indstart[0:nchunks]
            indsample = np.zeros(nchunks * chunklength)
            for istart in range(nchunks):
                indsample[(istart) * chunklength:(istart + 1) * chunklength] = np.arange(indstart[istart],
                                                                                         indstart[istart] + chunklength)

        indsample = indsample.astype('int32')
        # print(indsample)
        # print(indvar)
        # print(X.shape, Y.shape, indsample.shape)

        Y = Y[indsample]
        X = X[indsample, :]
        X = X[:, indvar]
        N, M = X.shape

        n = N - self.order
        X0 = np.zeros((n, self.order, M))
        Y0 = np.zeros((n, self.order))
        y = np.array([Y[self.order:]])
        for i in range(n):
            for j in range(self.order):
                Y0[i, j] = Y[self.order - j + i - 1]
                for k in range(M):
                    X0[i, j, k] = X[self.order - j + i - 1, k]

        X0_reshaped = np.reshape(np.ravel(X0, order='F'), (n, self.order * M), order='F').T
        Y0 = Y0.T
        # print(X0_reshaped.shape)
        # print(y.shape)
        # print(Y0.shape)
        o = - (M - 1) * self.get_cmi(y, X0_reshaped, Y0)
        # print(o)
        for k in range(M):
            X = np.delete(X0, k, axis=2)
            X_reshaped = np.reshape(np.ravel(X, order='F'), (n, self.order * (M - 1)), order='F').T
            o = o + self.get_cmi(y, X_reshaped, Y0)
        # print(o)
        # sys.exit()
        return o

    def run(self, ts, config):
        higher_order = config["higher_order"]
        Xfull = copnorm(ts)
        n_variables, n_timepoints = Xfull.shape
        print("Timeseries details - Number of variables: ", str(n_variables), ", Number of timepoints: ",
              str(n_timepoints))
        print("Computing dOinfo using " + self.estimator.type + " estimator")
        X = Xfull.T
        maxsize = config["maxsize"]  # 4 # max number of variables in the multiplet
        n_best = config["n_best"]  # 10 # number of most informative multiplets retained
        nboot = config["nboot"]  # 100 # number of bootstrap samples

        alphaval = 0.05
        o_b = np.zeros((nboot, 1))

        # this section is for the expansion of redundancy, so maximizing the O

        Odict = {}
        bar_length = n_variables * (maxsize + 1 - 2)
        with tqdm(total=bar_length) as pbar:
            pbar.set_description("Outer loops, maximizing O")
            for index_target in tqdm(range(n_variables), disable=True):
                Otarget = {}
                t = X[:, index_target]
                for isize in tqdm(range(2, maxsize + 1), disable=True):
                    Otot = {}
                    var_arr = np.setdiff1d(np.arange(1, n_variables + 1), index_target + 1)
                    if higher_order:
                        H = CombinationsManager(len(var_arr), isize)
                        ncomb = ncr(len(var_arr), isize)
                        O_pos = np.zeros(n_best)
                        O_neg = np.zeros(n_best)
                        ind_pos = np.zeros(n_best)
                        ind_neg = np.zeros(n_best)
                    else:
                        nplets_iter = itertools.combinations(var_arr, isize)
                        nplets = []
                        for nplet in nplets_iter:
                            nplets.append(nplet)
                        C = np.array(nplets)  # n-tuples without repetition over N modules
                        ncomb = C.shape[0]
                    Osize = np.zeros(ncomb)
                    for icomb in tqdm(range(ncomb), desc="Inner loop, computing O", leave=False):
                        if higher_order:
                            comb = H.nextchoose()
                            Osize = self.compute_dynamic_o(t, X, np.arange(n_timepoints),
                                                                   var_arr[comb - 1] - 1)
                            valpos, ipos = np.min(O_pos), np.argmin(O_pos)
                            valneg, ineg = np.max(O_neg), np.argmax(O_neg)
                            if Osize > 0 and Osize > valpos:
                                O_pos[ipos] = Osize
                                ind_pos[ipos] = H.combination2number(comb)
                            if Osize < 0 and Osize < valneg:
                                O_neg[ineg] = Osize
                                ind_neg[ineg] = H.combination2number(comb)
                        else:
                            comb = C[icomb, :]
                            Osize[icomb] = self.compute_dynamic_o(t, X, np.arange(n_timepoints),
                                                                          comb - 1)
                    if higher_order:
                        Osort_pos, ind_pos_sort = np.sort(O_pos)[::-1], np.argsort(O_pos)[::-1]
                        Osort_neg, ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
                    else:
                        ind_pos = np.argwhere(Osize > 0)
                        ind_neg = np.argwhere(Osize < 0)
                        O_pos = Osize[Osize > 0]
                        O_neg = Osize[Osize < 0]
                        Osort_pos, ind_pos_sort = np.sort(O_pos)[::-1], np.argsort(O_pos)[::-1]
                        Osort_neg, ind_neg_sort = np.sort(O_neg), np.argsort(O_neg)
                    if Osort_pos.size != 0:
                        n_sel = min(n_best, len(Osort_pos))
                        boot_sig = np.zeros((n_sel, 1))
                        for isel in range(n_sel):
                            if higher_order:
                                indvar = H.number2combination(ind_pos[ind_pos_sort[isel]])
                                f = lambda xsamp: self.o_information_lagged_boot(t, X, xsamp, self.chunk_length,
                                                                                 var_arr[indvar - 1] - 1)
                            else:
                                indvar = np.squeeze(C[ind_pos[ind_pos_sort[isel]], :])
                                f = lambda xsamp: self.o_information_lagged_boot(t, X, xsamp, self.chunk_length,
                                                                                 indvar - 1)
                            ci_lower, ci_upper = bootci(nboot, f, np.arange(n_timepoints - self.chunk_length + 1),
                                                        alphaval)
                            boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                        Otot['sorted_red'] = Osort_pos[0:n_sel]
                        Otot['index_red'] = ind_pos[ind_pos_sort[0:n_sel]].flatten()
                        Otot['bootsig_red'] = boot_sig
                    if Osort_neg.size != 0:
                        n_sel = min(n_best, len(Osort_neg))
                        boot_sig = np.zeros((n_sel, 1))
                        for isel in range(n_sel):
                            if higher_order:
                                indvar = H.number2combination(ind_neg[ind_neg_sort[isel]])
                                f = lambda xsamp: self.o_information_lagged_boot(t, X, xsamp, self.chunk_length,
                                                                                 var_arr[indvar - 1] - 1)
                            else:
                                indvar = np.squeeze(C[ind_neg[ind_neg_sort[isel]], :])
                                f = lambda xsamp: self.o_information_lagged_boot(t, X, xsamp, self.chunk_length,
                                                                                 indvar - 1)
                            ci_lower, ci_upper = bootci(nboot, f, np.arange(n_timepoints - self.chunk_length + 1),
                                                        alphaval)
                            boot_sig[isel] = not (ci_lower <= 0 and ci_upper > 0)
                        Otot['sorted_syn'] = Osort_neg[0:n_sel]
                        Otot['index_syn'] = ind_neg[ind_neg_sort[0:n_sel]].flatten()
                        Otot['bootsig_syn'] = boot_sig
                    Otot['var_arr'] = var_arr
                    # In case of higher_order = True,
                    # to get back the combination of the max Redundancy do -
                    # var_arr[H.number2combination(H.Otot['index_red'][0])-1]
                    # more details in read_outputs.py
                    Otarget[isize] = Otot
                    pbar.update(1)
                Odict[index_target] = Otarget
        return Odict
