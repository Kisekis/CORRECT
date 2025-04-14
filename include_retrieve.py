import os
from pathlib import Path
import re
from collections import deque
from cflow_prefilter import process_cflow_repo, subgraph_callee

include_pattern = re.compile(r'#include\s*[<"](.+?)[>"]')


def find_all_files(project_path):
    project_files = []
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith((".c", ".h")):
                full_path = Path(root) / file
                project_files.append(full_path)
    return project_files


def extract_includes(file_path, project_files):
    includes = set()
    file_dir = file_path.parent
    project_root = Path(os.path.commonpath([str(p) for p in project_files]))

    project_files_set = {str(p) for p in project_files}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = include_pattern.search(line)
                if match:
                    included_file = match.group(1)
                    found = False

                    local_path = os.path.normpath(str(file_dir / included_file))
                    if local_path in project_files_set:
                        includes.add(Path(local_path))
                        found = True
                    if not found:
                        root_path = os.path.normpath(str(project_root / included_file))
                        if root_path in project_files_set:
                            includes.add(Path(root_path))
                            found = True
                    if not found:
                        included_path_parts = Path(included_file).parts

                        for proj_file in project_files:
                            proj_parts = proj_file.parts

                            if (
                                len(proj_parts) >= len(included_path_parts)
                                and proj_parts[-len(included_path_parts) :]
                                == included_path_parts
                            ):
                                includes.add(proj_file)
                                found = True

                    if found and Path(local_path).suffix == ".h":

                        potential_c_file = Path(local_path).with_suffix(".c")
                        if str(potential_c_file) in project_files_set:
                            includes.add(potential_c_file)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return includes


def build_dependency_graph(target_file, project_files):
    dependency_graph = {}
    visited = set()
    queue = deque([(target_file, 0)])
    max_depth = 1

    while queue:
        current_file, depth = queue.popleft()
        if current_file in visited or depth > max_depth:
            continue
        visited.add(current_file)
        if current_file.suffix == ".h":
            c_file = current_file.with_suffix(".c")
            if c_file.exists():
                if c_file not in visited:
                    queue.append((c_file, depth))
                    includes = extract_includes(c_file, project_files)
                    dependency_graph[c_file] = includes

        includes = extract_includes(current_file, project_files)
        dependency_graph[current_file] = includes

        for include in includes:
            if include not in visited:
                queue.append((include, depth + 1))

    return dependency_graph


def find_reachable_files(project_path, target_file):
    project_files = find_all_files(project_path)

    target_file_path = Path(target_file).resolve()
    dependency_graph = build_dependency_graph(target_file_path, project_files)

    reachable_files = set(dependency_graph.keys())
    for includes in dependency_graph.values():
        reachable_files.update(includes)

    return reachable_files


def get_def_from_ctags(project_path, identifiers):
    tags_file = Path(project_path) / "tags"
    os.system(
        f"cd {project_path} && ctags -R --fields=+n-k-f-s --c++-kinds=+p --extras=+q ."
    )

    definition_files = set()

    try:
        with open(tags_file, "r", encoding="utf-8") as f:
            for line in f:

                if line.startswith("!"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    symbol = parts[0]
                    if symbol in identifiers:
                        file_path = Path(project_path) / parts[1]
                        definition_files.add(file_path)

        os.remove(tags_file)

    except Exception as e:
        print(f"Error processing tags file: {e}")
        if tags_file.exists():
            os.remove(tags_file)

    return definition_files


def prefilter(target_file, identifiers, project_path):
    ret = process_cflow_repo(project_path)
    new_identifiers = [f"{id}()" for id in identifiers]
    callee = subgraph_callee(ret, target_file, new_identifiers, 2)
    paths = set()
    for entry in callee:
        paths.add(entry["file_path"])
    return list(paths)


def get_call_graph_from_cflow(project_path, target_file, identifiers):
    related_files = set()
    try:
        paths = prefilter(target_file, identifiers, project_path)
        paths = [os.path.abspath(path) for path in paths]
        paths = [Path(path).as_posix() for path in paths]
        related_files.update(paths)
    except Exception as e:
        print(f"Error generating call graph: {e}")
    return related_files


def get_all_files(project_path, target_file, identifiers):
    reachable_files = find_reachable_files(project_path, target_file)
    definition_files = get_call_graph_from_cflow(project_path, target_file, identifiers)
    reachable_files.update(definition_files)
    return reachable_files
