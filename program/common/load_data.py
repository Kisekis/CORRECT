import os
import json
from .data_classes import (
    VulnPair,
    VulnContext,
    VulnPairWithContext,
    CalleeMethod,
    TypeDef,
    VulnMethod,
)
import pandas as pd
from common.logger import logger


def vulnerable_method_to_str(methods):
    ret = ""
    for method in methods:
        ret += f"//filename:{method[0]}\n{method[2]}\n"
    return ret


def construct_context(data: dict) -> VulnContext:
    """
    Constructs a VulnContext object from a dictionary of data.

    Args:
        data: Dictionary containing context information

    Returns:
        VulnContext object with properly typed data
    """

    callee_methods = [
        CalleeMethod(
            filename=method[0],
            method_name=method[1],
            raw_code=method[2],
            depth=method[3],
        )
        for method in data.get("calleeMethods", [])
    ]

    type_defs = [
        TypeDef(type_def=type_def[0], name=type_def[1])
        for type_def in data.get("typeDefs", [])
    ]

    visited_params = {}
    for param_list in data.get("visitedParams", {}).values():
        for param, method in param_list:
            if method not in visited_params:
                visited_params[method] = []
            if param not in visited_params[method]:
                visited_params[method].append(param)

    return VulnContext(
        calleeMethods=callee_methods,
        typeDefs=type_defs,
        globalVars=data.get("globalVars", []),
        importContext=data.get("importContext", []),
        visitedLines_before=data.get("visitedLines_before", {}),
        visitedLines_after=data.get("visitedLines_after", {}),
        visitedParams=visited_params,
    )


def load_pair_with_more_context(meta_data_path, file_path, project, hash):
    logger.info(f"meta_data_path: {meta_data_path}, file_path: {file_path}")
    all_data = pd.read_csv(meta_data_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    row = all_data[
        (all_data["repo_name"].str.endswith(project))
        & (all_data["hash"].str.startswith(hash))
    ]
    if row.empty:
        return None
    row = row.iloc[0]
    logger.info(f"row: {row}")

    vuln_methods = [
        VulnMethod(
            filename=method[0],
            method_name=method[1],
            raw_code=method[2],
            start_line=method[3],
        )
        for method in data["vulnerableMethods_before"]
    ]
    patched_methods = [
        VulnMethod(
            filename=method[0],
            method_name=method[1],
            raw_code=method[2],
            start_line=method[3],
        )
        for method in data["vulnerableMethods_after"]
    ]

    cwe_list = (
        row["cwe_list"].replace("'", "").replace('"', "").replace(" ", "").split(",")
    )
    if "NVD-CWE-noinfo" in cwe_list:
        cwe_list.remove("NVD-CWE-noinfo")
    return VulnPairWithContext(
        cwe=cwe_list,
        vuln=vuln_methods,
        patched=patched_methods,
        name=hash,
        context=construct_context(data),
    )


def load_pairs_from_csv(file_path):

    pairs = []
    df = pd.read_csv(file_path)

    for _, entry in df.iterrows():
        pair = VulnPair(
            cwe=entry["cwe_id"],
            vuln=entry["code_before"],
            patched=entry["code_after"],
            name=entry["fix_hash"][:8],
            method=entry["method_name"],
        )
        pairs.append(pair)

    return pairs


def load_pairs(base_dir, cwe):
    """
    Load vulnerability and patched code pairs from the specified directory.

    Args:
        base_dir (str): The base directory containing CWE folders.
        cwe (str): The specific CWE folder to process.

    Returns:
        list[VulnPair]: A list of VulnPair objects with loaded code.
    """
    cwe_upper = cwe.upper()
    cwe_dir = os.path.join(base_dir, cwe_upper)
    if not os.path.exists(cwe_dir):
        raise FileNotFoundError(f"Directory {cwe_dir} does not exist.")

    pairs = []
    files = os.listdir(cwe_dir)
    idx_set = set(
        f.split(".")[0] for f in files if f.endswith(".c") and not f.endswith("_p.c")
    )

    for idx in idx_set:
        vuln_path = os.path.join(cwe_dir, f"{idx}.c")
        patched_path = os.path.join(cwe_dir, f"{idx}_p.c")

        if not os.path.exists(patched_path):
            raise FileNotFoundError(
                f"Patched file {patched_path} for {idx}.c is missing."
            )

        with open(vuln_path, "r", encoding="utf-8") as f:
            vuln_code = f.read()

        with open(patched_path, "r", encoding="utf-8") as f:
            patched_code = f.read()

        pairs.append(
            VulnPair(cwe=cwe_upper, vuln=vuln_code, patched=patched_code, name=idx)
        )

    return pairs
