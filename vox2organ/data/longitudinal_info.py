import collections.abc as abc
import logger
from enum import IntEnum, auto
from collections import OrderedDict

import pandas as pd
from typing import List, Optional, Dict, Any, Sequence
import itertools

from utils.template import TEMPLATE_SPECS
from utils.modes import TemplateModes


log = logger.get_std_logger(__name__)



# function decorator to make sure that the object is loaded
def require_load(func):
    def wrapper(self, *args, **kwargs):
        if not self.loaded:
            self.load()
        return func(self, *args, **kwargs)
    return wrapper


class LongitudinalInfo:

    NECESSARY_COLS = ["PTID", "SCAN_NO"]
    OPTIONAL_COLS = ["SPLIT", "MONTHS", "OVERALLQC", "PTGENDER"]

    def __init__(self, csv_path, scan_col="IMAGEUID", splits_filter=("train", "val", "test")):
        self.path = csv_path
        self.splits_filter = splits_filter
        self.scan_col = scan_col
        self.single_scans = not isinstance(csv_path, str)  or csv_path.endswith(".txt")
        self.overfit = None

        self.loaded = False


    def load(self):
        if self.loaded: return


        if not self.single_scans:
            self.df = pd.read_csv(self.path)

            # check if necessary columns are present
            assert(all([col in self.df.columns for col in self.NECESSARY_COLS]))

            # double check quality control
            if "OVERALLQC" in self.df.columns:
                assert(self.df["OVERALLQC"].unique() == ["Pass"]).all()

            # filter df by split column
            if "SPLIT" in self.df.columns:
                self.df = self.df[self.df["SPLIT"].isin(self.splits_filter)]
        else:
            if isinstance(self.path, str):
                self.df = pd.read_csv(self.path, sep=" ", header=None, names=[self.scan_col])
            else:
                self.df = pd.DataFrame({self.scan_col: self.path})
            self.df["PTID"] = [f"single_scan{x}" for x in self.df.index]
            self.df[self.scan_col] = 0


        # convert scan no to string just in case
        self.df[self.scan_col] = self.df[self.scan_col].astype(str)
        # convert ptid to string just in case
        self.df["PTID"] = self.df["PTID"].astype(str)

        self.loaded = True
        self.sanity_check_df()


    def __repr__(self):
        return f"LongitudinalInfo({self.path}, {self.scan_col}, {self.splits_filter})"

    def set_overfit(self, overfit: int):
        """
        this will make get_mappings return max(overfit, len(self.df)) scans
        """
        assert overfit > 0
        self.overfit = overfit


    @require_load
    def dump(self, path):
        self.df.to_csv(path, index=False)

    @require_load
    def sanity_check_df(self):
        """
        most of these were generated using copilot
        """
        if "SPLIT" in self.df.columns:
            # check that PTID is consistend with split
            assert(self.df.groupby("PTID")["SPLIT"].apply(lambda x: x.unique().tolist())\
                    .apply(lambda x: len(x) == 1).all())

        if "SCAN_NO" in self.df.columns:
            # assert data type
            assert(self.df["SCAN_NO"].dtype == int)
            # assert per patient:
            #  * starts by 0
            #  * consecutively increasing
            assert(self.df.groupby("PTID")["SCAN_NO"].apply(lambda x: x.tolist())\
                    .apply(lambda x: x == list(range(len(x)))).all())
        if "MONTHS" in self.df.columns:
            assert(self.df["MONTHS"].dtype == float or self.df["MONTHS"].dtype == int)

        if "MONTHS" in self.df.columns and "SCAN_NO" in self.df.columns:
            # assert that months are never decreasing with increasing scan_no
            assert self.df.groupby("PTID").apply(lambda x: x.sort_values("SCAN_NO")["MONTHS"].tolist() == sorted(x["MONTHS"])).all()

        # check gender consistency
        if "PTGENDER" in self.df.columns:
            assert self.df.groupby("PTID")["PTGENDER"].apply(lambda x: x.unique().tolist())\
                    .apply(lambda x: len(x) == 1).all()

        if any(template_id in self.df[self.scan_col].tolist() for template_id in TEMPLATE_SPECS.keys()):
            raise ValueError(f"Naming conflict between template ids and scan ids: \
                            {[template_id for template_id in TEMPLATE_SPECS.keys() if template_id in self.df[self.scan_col].tolist()]}")

        # check that scan col is unique
        assert len(self.df[self.scan_col].unique()) == len(self.df[self.scan_col])




    @require_load
    def get_all_scans(self, with_patient=False):
        if with_patient:
            return self.df[["PTID", self.scan_col]]
        return self.df[self.scan_col]

    @require_load
    def get_patient_dict(self):
        return self.df.groupby("PTID")[self.scan_col].apply(list).to_dict()


    @require_load
    def get_mappings(self,
                     mode: TemplateModes,
                     static_template_id: Optional[str]=None,
                     additional_features: Sequence[str]=tuple()) -> OrderedDict[str, List[Dict[str, Any]]]:
        """
        Returns an ordered_dict: {ptid -> [dict("template" -> id, "scan" -> id, feature_x -> value, ...)]} for the given mode.
        if a static template is used, the ptid is set to "single_scanX" where X is an increasing number

        the additonal features may be returned:
            - "months_diff": the number of months between the scan and the template scan. Requires column "MONTHS"
            - "gender": 1 if female, 0 male, requires column "PTGENDER"
            - any other column name in the csv can be saved by passing it in additional_features.
              it will be prefixed by template_ or scan_ depending on the column it is from.

        """
        if static_template_id is not None and mode != TemplateModes.STATIC:
            log.warning("static_template_id is ignored for mode != TemplateModes.STATIC")

        if self.single_scans and mode != TemplateModes.STATIC and len(self.df) > 0:
            log.warning("single scans do not have any time information, any mode != TemplateModes.STATIC does not make sense")

        if "SCAN_NO" not in self.df.columns:
            groups = self.df.sort_values(self.scan_col).groupby("PTID")
        else:
            # groups sorted by SCAN_NO, but keep columns
            groups = self.df.sort_values("SCAN_NO").groupby("PTID")


        if mode == TemplateModes.STATIC:
            assert static_template_id is not None
            # # if additional_features:
            # #     log.warning("additional_features are ignored for mode == TemplateModes.STATIC")
            res_pairs = groups.apply(lambda x: [(static_template_id, scan) for scan in x[self.scan_col].tolist()])
        elif mode == TemplateModes.PREV:
            res_pairs = groups.apply(lambda x: zip(x[self.scan_col].tolist()[:-1], x[self.scan_col].tolist()[1:]))
        elif mode == TemplateModes.FIRST:
            res_pairs = groups.apply(lambda x: [(x[self.scan_col].tolist()[0], scan) for scan in x[self.scan_col].tolist()])
        elif mode == TemplateModes.NXN:
            res_pairs = groups.apply(lambda x: list(itertools.product(x[self.scan_col], repeat=2)))
        elif mode == TemplateModes.NXN_SORTED:
            res_pairs = groups.apply(lambda x: [(template, scan) for (template,scan) in itertools.combinations_with_replacement(x[self.scan_col], 2)])
        elif mode == TemplateModes.MEAN or mode == TemplateModes.MEDIAN:
            res_pairs = groups.apply(lambda x: [(x['PTID'].tolist()[0], scan) for scan in x[self.scan_col].tolist()])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        res = OrderedDict()
        cnt = 0
        for ptid, pairs in res_pairs.items():
            res_pt = []
            for template, scan in pairs:
                res_pair = {"patient": ptid, "template": template, "scan": scan}
                if "MONTHS" in self.df.columns and mode not in [TemplateModes.STATIC, TemplateModes.MEAN, TemplateModes.MEDIAN]:
                    res_pair["months_diff"] =   self.df[self.df[self.scan_col] == scan]["MONTHS"].values[0] \
                                              - self.df[self.df[self.scan_col] == template]["MONTHS"].values[0]
                elif "MONTHS" in self.df.columns and mode in [TemplateModes.MEAN, TemplateModes.MEDIAN]:
                    # compute mean months per patient
                    mean_month = self.df[self.df["PTID"] == ptid]["MONTHS"].mean()
                    res_pair["months_diff"] =   self.df[self.df[self.scan_col] == scan]["MONTHS"].values[0] \
                                                - mean_month
                    # TODO: use median month for median template?
                elif "months_diff" in additional_features:
                    raise ValueError("additional_features contains months_diff, but MONTHS column is not present in csv")


                if "PTGENDER" in self.df.columns:
                    res_pair["gender"] = 1 if self.df[self.df[self.scan_col] == scan]["PTGENDER"].values[0] == "Female" else 0
                elif "gender" in additional_features:
                    raise ValueError("additional_features contains gender, but PTGENDER column is not present in csv")

                for additional_feature in additional_features:
                    if additional_feature in ["months_diff", "gender"]:
                        continue
                    elif additional_feature.startswith("template_"):
                        res_pair[additional_feature] = self.df[self.df[self.scan_col] == template][additional_feature[9:]].values[0]
                    elif additional_feature.startswith("scan_"):
                        res_pair[additional_feature] = self.df[self.df[self.scan_col] == scan][additional_feature[5:]].values[0]
                    elif additional_feature in self.df.columns:
                        res_pair[f"template_{additional_feature}"] = self.df[self.df[self.scan_col] == template][additional_feature].values[0]
                        res_pair[f"scan_{additional_feature}"] = self.df[self.df[self.scan_col] == scan][additional_feature].values[0]
                    else:
                        log.warning(f"Additional feature {additional_feature} is not in the columns of the csv. \
                                    Skipping this feature.")

                # add scan number
                if "SCAN_NO" in self.df.columns:
                    res_pair["scan_no"] = self.df[self.df[self.scan_col] == scan]["SCAN_NO"].values[0]
                else:
                    res_pair["scan_no"] = None

                res_pt.append(res_pair)
                cnt += 1
                if self.overfit is not None and cnt >= self.overfit:
                    break

            res[ptid] = res_pt
            if self.overfit is not None and cnt >= self.overfit:
                break

        # from IPython import embed; embed()

        return res
