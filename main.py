import os
import subprocess
import json
from patch_analyzer import analyze_patch
from include_retrieve import get_all_files
import pandas as pd
import signal
from functools import wraps
import time
import shutil
import csv
import argparse


def get_parent_commit(repo_path: str, commit_hash: str) -> str:
    """Get the parent commit hash of the given commit"""
    result = subprocess.run(
        ["git", "rev-parse", f"{commit_hash}^"],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )
    return result.stdout.strip()


def checkout_commit(repo_path: str, commit_hash: str):
    """Checkout to specific commit"""
    subprocess.run(["git", "checkout", commit_hash, "-f"], check=True, cwd=repo_path)


def create_exclude_regex(repo_path: str, file_path: str, changed_methods: set):
    repo_path = os.path.abspath(repo_path)
    file_path = os.path.abspath(os.path.join(repo_path, file_path))
    reachable_files = get_all_files(repo_path, file_path, changed_methods)

    reachable_relative_paths = {
        os.path.relpath(file, repo_path) for file in reachable_files
    }

    return f"^(?!({'|'.join(reachable_relative_paths)})$).*\\.(c|h)$"


def map_line_numbers(file_patches, file_path, visited_lines, version):

    if file_path not in file_patches:
        raise ValueError(f"Patches for {file_path} not found!")

    patch = file_patches[file_path]
    del_lines = set(patch.get("del_lines", []))
    add_lines = set(patch.get("add_lines", []))

    del_lines = sorted(set([int(line) for line in del_lines]))
    add_lines = sorted(set([int(line) for line in add_lines]))

    mapped_lines = []

    if version == "before":

        for line in visited_lines:
            if line in del_lines:
                pass
            else:

                offset = -sum(1 for d in del_lines if d < line) + sum(
                    1 for a in add_lines if a < line
                )
                mapped_lines.append(line + offset)

    elif version == "after":

        for line in visited_lines:
            if line in add_lines:
                pass
            else:

                offset = sum(1 for a in add_lines if a < line) - sum(
                    1 for d in del_lines if d < line
                )
                mapped_lines.append(line - offset)
    else:
        raise ValueError("wrong version")

    return mapped_lines


def run_joern_analysis(repo_path: str, commit_hash: str):

    cpgs_dir = os.path.join(os.getcwd(), "cpgs")
    contexts_dir = os.path.join(os.getcwd(), "contexts_new")
    os.makedirs(cpgs_dir, exist_ok=True)
    os.makedirs(contexts_dir, exist_ok=True)

    parent_hash = get_parent_commit(repo_path, commit_hash)

    patches = analyze_patch(repo_path, commit_hash)

    contexts = {"before": None, "after": None}

    changed_methods = {}
    callee_methods = set()

    file_patches = {}

    for patch in patches:
        if patch.file_path not in file_patches:
            file_patches[patch.file_path] = {"del_lines": [], "add_lines": []}
        file_patches[patch.file_path]["del_lines"].extend(patch.deleted_lines.keys())
        file_patches[patch.file_path]["add_lines"].extend(patch.added_lines.keys())

    for version, version_hash in [("before", parent_hash), ("after", commit_hash)]:
        checkout_commit(repo_path, version_hash)

        for file_path, file_patches_list in file_patches.items():
            if version == "before":
                line_numbers = file_patches_list["del_lines"]
            else:
                line_numbers = file_patches_list["add_lines"]

            if not line_numbers:
                line_numbers = [0]

            line_numbers_str = ",".join(map(str, sorted(line_numbers)))

            methods_output = os.path.join(
                cpgs_dir, f"methods_{version}_{os.path.basename(file_path)}.txt"
            )

            cmd = [
                "joern",
                "--script",
                "get_methods.scala",
                "--param",
                f"codeFile={os.path.join(repo_path, file_path)}",
                "--param",
                f"outFile={methods_output}",
                "--param",
                f"lineNumbers={line_numbers_str}",
            ]
            subprocess.run(cmd, check=True)

            if os.path.exists(methods_output):
                with open(methods_output, "r") as f:
                    for line in f:
                        method_info = line.strip().split()
                        if method_info:
                            method_name = method_info[0]
                            callees = method_info[1:] if len(method_info) > 1 else []

                            if file_path not in changed_methods:
                                changed_methods[file_path] = set()
                            changed_methods[file_path].add(method_name)

                            callee_methods.update(callees)

                os.remove(methods_output)

    if len(changed_methods) == 0:
        return
    changed_methods_list = []
    for file_path, methods in changed_methods.items():
        changed_methods_list.extend(methods)

    for version, version_hash in [("before", parent_hash), ("after", commit_hash)]:
        checkout_commit(repo_path, version_hash)

        base_name = f"{os.path.basename(repo_path)}_{version_hash[:8]}"
        cpg_path = os.path.join(cpgs_dir, f"{base_name}.bin")

        temp_dir = os.path.join(contexts_dir, f"temp_{version_hash}")
        os.makedirs(temp_dir, exist_ok=True)
        generated_json_files = []

        for file_path in file_patches.keys():
            subprocess.run(
                [
                    "joern-parse",
                    ".",
                    "-o",
                    cpg_path,
                    "--language",
                    "C",
                    "--frontend-args",
                    "--exclude-regex",
                    create_exclude_regex(repo_path, file_path, changed_methods_list),
                ],
                check=True,
                cwd=repo_path,
            )

        for file_path, methods in changed_methods.items():
            out_file = f"context_{version}_{os.path.basename(file_path)}.json"
            out_file_path = os.path.join(temp_dir, out_file)
            generated_json_files.append(out_file_path)

            if version == "before":
                line_numbers = file_patches[file_path]["del_lines"]
            else:
                line_numbers = file_patches[file_path]["add_lines"]

            if not line_numbers:
                line_numbers = [0]

            line_numbers_str = ",".join(map(str, sorted(line_numbers)))

            cmd = [
                "joern",
                "--script",
                "bfs.scala",
                "--param",
                f'methodnameList={",".join(changed_methods[file_path])}',
                "--param",
                f"cpgFile={cpg_path}",
                "--param",
                f"outFile={out_file_path}",
                "--param",
                f"filename={file_path}",
                "--param",
                f"lineNumbers={line_numbers_str}",
            ]
            subprocess.run(cmd, check=True)

        merged_data = {
            "calleeMethods": [],
            "typeDefs": [],
            "globalVars": [],
            "importContext": [],
            "vulnerableMethods": [],
            "visitedLines": [],
            "visitedParams": [],
        }

        for json_file in generated_json_files:
            if os.path.exists(json_file):
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key in merged_data:
                        merged_data[key].extend(data.get(key, []))

        contexts[version] = merged_data

        shutil.rmtree(temp_dir)

    final_context = {
        "calleeMethods": [],
        "typeDefs": [],
        "globalVars": [],
        "importContext": [],
        "vulnerableMethods_before": [],
        "vulnerableMethods_after": [],
        "visitedLines_before": [],
        "visitedLines_after": [],
        "visitedParams": [],
    }

    all_callee_methods = set(
        tuple(item)
        for item in contexts["before"]["calleeMethods"]
        + contexts["after"]["calleeMethods"]
    )
    final_context["calleeMethods"] = [
        list(item)
        for item in all_callee_methods
        if (
            not (
                item[0] in changed_methods.keys()
                and item[1] in changed_methods[item[0]]
            )
        )
        and item[0] != "<empty>"
    ]

    all_typedefs = set(
        tuple(item)
        for item in contexts["before"]["typeDefs"] + contexts["after"]["typeDefs"]
    )
    final_context["typeDefs"] = [list(item) for item in all_typedefs]

    final_context["globalVars"] = list(
        set(contexts["before"]["globalVars"] + contexts["after"]["globalVars"])
    )
    final_context["importContext"] = list(
        set(contexts["before"]["importContext"] + contexts["after"]["importContext"])
    )

    final_context["vulnerableMethods_before"] = contexts["before"]["vulnerableMethods"]
    final_context["vulnerableMethods_after"] = contexts["after"]["vulnerableMethods"]

    visitedLines_before = {}
    for entry in contexts["before"]["visitedLines"]:
        line_number, method_name, file_path = entry
        if file_path not in visitedLines_before:
            visitedLines_before[file_path] = []
        visitedLines_before[file_path].append(line_number)

    visitedLines_after = {}
    for entry in contexts["after"]["visitedLines"]:
        line_number, method_name, file_path = entry
        if file_path not in visitedLines_after:
            visitedLines_after[file_path] = []
        visitedLines_after[file_path].append(line_number)

    before_to_after = {}
    after_to_before = {}
    for file_path in visitedLines_before:
        before_to_after[file_path] = map_line_numbers(
            file_patches, file_path, visitedLines_before[file_path], "before"
        )

    for file_path in visitedLines_after:
        after_to_before[file_path] = map_line_numbers(
            file_patches, file_path, visitedLines_after[file_path], "after"
        )

    for file_path in before_to_after:
        before_to_after[file_path].extend(visitedLines_after[file_path])

    for file_path in after_to_before:
        after_to_before[file_path].extend(visitedLines_before[file_path])

    for file_path in after_to_before:
        after_to_before[file_path] = list(set(after_to_before[file_path]))
        after_to_before[file_path] = sorted(after_to_before[file_path])

    for file_path in before_to_after:
        before_to_after[file_path] = list(set(before_to_after[file_path]))
        before_to_after[file_path] = sorted(before_to_after[file_path])

    final_context["visitedLines_before"] = after_to_before
    final_context["visitedLines_after"] = before_to_after

    file_method_params = {}

    for param, method_name, file_path in contexts["before"]["visitedParams"]:
        if file_path not in file_method_params:
            file_method_params[file_path] = set()
        file_method_params[file_path].add((param, method_name))

    for param, method_name, file_path in contexts["after"]["visitedParams"]:
        if file_path not in file_method_params:
            file_method_params[file_path] = set()
        file_method_params[file_path].add((param, method_name))

    for file_path in file_method_params:
        file_method_params[file_path] = sorted(file_method_params[file_path])

    final_context["visitedParams"] = file_method_params

    context_json_path = os.path.join(
        contexts_dir, f"{os.path.basename(repo_path)}_{commit_hash[:8]}_context.json"
    )

    def ordered(obj):
        if isinstance(obj, dict):
            return sorted((k, ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(ordered(x) for x in obj)
        else:
            return obj

    with open(context_json_path, "w", encoding="utf-8") as f:
        json.dump(final_context, f, indent=2)

    checkout_commit(repo_path, commit_hash)


def run_analysis(repo, commit_hash):
    repo_path = f"projects/{repo}"
    run_joern_analysis(repo_path, commit_hash)


def timeout_handler(signum, frame):
    raise TimeoutError("Analysis timed out after 2 minutes")


def timeout_decorator(seconds=120):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


@timeout_decorator(60 * 10)
def run_analysis_with_timeout(repo, commit_hash):
    return run_analysis(repo, commit_hash)


def run_analysis_for_all_entries():
    data_path = "data/CIVDataset-metadata.csv"
    df = pd.read_csv(data_path)
    count = 0
    for index, row in df.iterrows():
        count += 1
        if count < 271:
            continue
        repo = row["repo_name"].split("/")[1]
        commit_hash = row["hash"]
        file_name = f"{repo}_{commit_hash[:8]}_context.json"
        file_path = os.path.join("CIVDataset", file_name)
        if os.path.exists(file_path):
            continue
        if repo != "linux":
            continue

        start_time = time.time()

        try:
            print(f"Running analysis for {repo} with commit hash {commit_hash}")
            run_analysis_with_timeout(repo, commit_hash)
            duration = time.time() - start_time
            print(f"Analysis completed in {duration:.2f} seconds")

        except TimeoutError:
            print(f"Analysis for {repo} ({commit_hash}) timed out after 2 minutes")
            exit(0)
        except Exception as e:
            print(
                f"Error running analysis for {repo} with commit hash {commit_hash}: {e}"
            )


if __name__ == "__main__":
    run_analysis_for_all_entries()
