import re
from collections import defaultdict, deque


def parse_cflow_output(cflow_output):
    stack = []
    function_map = defaultdict(list)
    function_regex = re.compile(r"^\s*(\w+\(\))\s*<.*?>:")
    callee_regex = re.compile(r"^\s*(\w+\(\))")

    for line in cflow_output.splitlines():
        indent_level = len(line) - len(line.lstrip())
        while stack and stack[-1][1] >= indent_level:
            stack.pop()

        function_match = function_regex.match(line)
        if function_match:
            current_function = function_match.group(1)
            if stack:
                parent_function = stack[-1][0]
                if current_function not in function_map[parent_function]:
                    function_map[parent_function].append(current_function)
            stack.append((current_function, indent_level))
        else:
            callee_match = callee_regex.match(line)
            if callee_match and stack:
                parent_function = stack[-1][0]
                callee_function = callee_match.group(1)
                if callee_function not in function_map[parent_function]:
                    function_map[parent_function].append(callee_function)

    return dict(function_map)


def process_cflow_file(file_path):
    import subprocess

    result = subprocess.run(["cflow", file_path], capture_output=True, text=True)
    if result.returncode == 0:
        return parse_cflow_output(result.stdout)
    else:
        raise RuntimeError(f"Error processing {file_path}: {result.stderr}")


def process_cflow_repo(repo_path):
    import os

    repo_map = {}

    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".c") or file.endswith(".h"):
                file_path = os.path.join(root, file)
                try:
                    repo_map[file_path] = process_cflow_file(file_path)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    return repo_map


def subgraph_callee(graph_dict, file_path, functions, depth=2):
    callee = []
    queue = deque([(file_path, func, 0) for func in functions])
    visited = set()

    while queue:
        current_file, current_func, current_depth = queue.popleft()
        if current_depth > depth:
            continue

        key = (current_file, current_func)
        if key in visited:
            continue
        visited.add(key)

        callee.append(
            {
                "file_path": current_file,
                "function": current_func,
                "depth": current_depth,
            }
        )

        if current_file in graph_dict and current_func in graph_dict[current_file]:
            for callee_func in graph_dict[current_file][current_func]:
                queue.append((current_file, callee_func, current_depth + 1))

        for other_file, functions_map in graph_dict.items():
            if other_file != current_file and current_func in functions_map:
                queue.append((other_file, current_func, current_depth + 1))

    return callee
