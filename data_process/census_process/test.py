import numpy as np
import pandas as pd


def Value_counts(das, nhead=5):
    #     tmp = pd.value_counts(das).reset_index().rename_axis({"index": das.name}, axis = 1)
    tmp = pd.value_counts(das).reset_index().rename({"index": das.name}, axis=1)
    value = pd.DataFrame(['value {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)) \
        .join(tmp.iloc[:, 0], how="left").set_index(0).T
    freq = pd.DataFrame(['freq {}'.format(x + 1) for x in range(nhead)], index=np.arange(nhead)) \
        .join(tmp.iloc[:, 1], how="left").set_index(0).T
    nnull = das.isnull().sum()
    freqother = pd.DataFrame({das.name: [das.shape[0] - nnull - np.nansum(freq.values), nnull]},
                             index=["freq others", "freq NA"]).T
    op = pd.concat([value, freq, freqother], axis=1)
    return (op)


def Summary(da):
    op = pd.concat([pd.DataFrame({"type": da.dtypes, "n": da.notnull().sum(axis=0)}), da.describe().T.iloc[:, 1:],
                    pd.concat(map(lambda i: Value_counts(da.loc[:, i]), da.columns))], axis=1).loc[da.columns]
    op.index.name = "Columns"
    return (op)


def Col_group(ic, i=0):
    cols = pd.Series([x.split("_")[i] for x in ic], index=ic)
    return (cols)


def Time_to_num(das):
    tmp = pd.DatetimeIndex(das)
    daop = pd.DataFrame(dict(
        zip(*[["{}_{}".format(das.name, i) for i in ["Day", "Month", "Year", "DayofYear", "DayofMonth", "DayofWeek"]],
              [(das - das.min()).astype('timedelta64[D]').astype(int), tmp.month, tmp.year, tmp.dayofyear, tmp.day,
               tmp.dayofweek]])),
        index=das.index)
    return (daop)


def Cat_map(das, damap, fillna={"CityRank": 6}):
    daop = das.reset_index().set_index([das.name]).join(damap, how="left").set_index(das.index.name).reindex(
        das.index).fillna(fillna)
    daop.columns = ["{}_{}".format(das.name, i) for i in damap.columns]
    return (daop)


# def Cat_to_bin(das, a=0.01):
#     return pd.get_dummies(das)


def Append_col_name(da, name):
    return (da.rename(columns=dict(zip(*[list(da.columns), ["{}_{}".format(x, name) for x in da.columns]]))))


def ColS_fillna(da, cols, f="median", allNA=0):
    dafill = getattr(da[cols.index].groupby(cols, axis=1), f)()[cols]
    dafill.columns = cols.index
    daop = da[cols.index].fillna(dafill).fillna(allNA)
    return (daop)


def ColS_summary(da, cols, f=["median", "std"]):
    grp = da[cols.index].groupby(cols, axis=1)
    daop = pd.concat(map(lambda x: Append_col_name(getattr(grp, x)(), x), f), axis=1)
    return (daop)


def Clean_data(da, ictype, a=0.01):
    """
        Transform and clean columns according to types
    """

    dac = da.copy().replace([-np.inf, np.inf], np.nan).replace("不详", np.nan)
    dac.loc[:, "UserInfo_20"] = dac.loc[:, "UserInfo_20"].fillna(dac.loc[:, "UserInfo_19"])
    datime = pd.concat(map(lambda i: Time_to_num(dac.loc[:, i]), ictype["date"]), axis=1)
    print("datime:", datime)
    dacatmap = pd.concat(
        map(lambda i: Cat_map(dac.loc[:, ictype["catmap"][i]], ictype["catmapd"][i]), range(len(ictype["catmap"]))),
        axis=1)
    #     print(dacatmap.iloc[:,15:20])
    dacatmap = pd.concat([dacatmap.iloc[:, 15:20], ColS_summary(dacatmap,
                                                                pd.Series(
                                                                    ["_".join([x.split("_")[i] for i in [0, 2]]) for x
                                                                     in dacatmap.columns[:15]],
                                                                    index=dacatmap.columns[:15]))], axis=1)

    #dacatbin  = pd.concat(map(lambda i: Cat_to_bin(dac.loc[:,i], a = a), ictype["catbin"]+[ictype["catmap"][-1]]), axis = 1)
    # dacatbin = pd.concat(map(lambda i: Cat_to_bin(dac[[i]], a=a), ictype["catbin"]), axis=1)
    daS = ColS_summary(dac, ictype["serials"], ["median", "std", "min", "max", "first"]).fillna(0)
    cols = Col_group(daS.columns, i=-1)
    daS.loc[:, cols == "max"] = daS.loc[:, cols == "max"] - daS.loc[:, cols == "median"].values
    dacount = ColS_summary(dac, ictype["cols"], ["count"])
    # dac = pd.concat(
    #     [dac.drop(ictype["date"] + ictype["catmap"] + ictype["catbin"] + list(ictype["serials"].index), axis=1),
    #      datime, dacatmap, dacatbin, daS, dacount], axis=1)
    dac = pd.concat(
        [dac.drop(ictype["date"] + ictype["catmap"] + ictype["catbin"] + list(ictype["serials"].index), axis=1),
         datime, dacatmap, daS, dacount], axis=1)
    tmp = pd.concat(map(lambda i: Value_counts(dac.loc[:, i]), dac.columns))
    dac = dac.loc[:, (tmp["freq 1"] + tmp["freq NA"]) / dac.shape[0] < 1 - a]
    #dac = dac.drop(dac.columns[np.any(np.abs(np.tril(np.corrcoef(dac.rank(pct = True).fillna(0.5).values, rowvar = 0), -1)) > 0.99, axis = 0)], axis = 1)
    return (dac)
