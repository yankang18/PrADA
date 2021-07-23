import numpy as np

from utils import compute

if __name__ == "__main__":
    # centralized w/ DA w/ FG and using src+tgt
    pdd_all_w_da_auc = np.array([0.7549, 0.7525, 0.7565, 0.7563, 0.7536])
    pdd_all_w_da_ks = np.array([0.4302, 0.4289, 0.4400, 0.4275, 0.4209])

    # centralized with w/o DA and w/o FG using src+tgt
    ppd_no_fg_no_da_all_auc = np.array([0.7479, 0.7437, 0.7526, 0.7385, 0.7522, 0.7485, 0.7514, 0.7503, 0.7552, 0.7477])
    ppd_no_fg_no_da_all_ks = np.array([0.4082, 0.4203, 0.4158, 0.4122, 0.4108, 0.4053, 0.4082, 0.4102, 0.4222, 0.4199])

    # centralized with w/o DA and w/o FG using tgt
    ppd_no_fg_no_da_tgt_auc = np.array([0.6430, 0.6370, 0.6431, 0.6485, 0.6536, 0.6505, 0.6399, 0.6302, 0.6408, 0.6523])
    ppd_no_fg_no_da_tgt_ks = np.array([0.2561, 0.2470, 0.2638, 0.2418, 0.2523, 0.2745, 0.2392, 0.2411, 0.2565, 0.2543])

    # PrADA w/o FG
    ppd_da_no_fg_auc = np.array(
        [0.7473, 0.7463, 0.7480, 0.7433, 0.7438, 0.7437, 0.7562, 0.7506, 0.7515, 0.7441, 0.7549])
    ppd_da_no_fg_ks = np.array(
        [0.4126, 0.4092, 0.4044, 0.3958, 0.4057, 0.4109, 0.4112, 0.4132, 0.4093, 0.3920, 0.40592])

    # PrADA
    ppd_fg_da_auc = np.array([0.7520, 0.7473, 0.7493, 0.7563, 0.7526, 0.7454, 0.7515, 0.7536, 0.7543, 0.7511, 0.7516])
    ppd_fg_da_ks = np.array([0.4280, 0.4151, 0.4143, 0.4284, 0.4187, 0.4248, 0.4093, 0.4140, 0.4269, 0.4258, 0.4110])

    compute(pdd_all_w_da_auc, pdd_all_w_da_ks, "all_w_da")
    compute(ppd_no_fg_no_da_all_auc, ppd_no_fg_no_da_all_ks, "no_fg_no_da_all")
    compute(ppd_no_fg_no_da_tgt_auc, ppd_no_fg_no_da_tgt_ks, "no_fg_no_da_tgt")
    compute(ppd_da_no_fg_auc, ppd_da_no_fg_ks, "da_no_fg")
    compute(ppd_fg_da_auc, ppd_fg_da_ks, "fg_da")
