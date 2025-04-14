import pandas as pd

df = pd.read_csv("../data/1000.csv", index_col=False)


class CWEGraph:
    def __init__(self):
        self.nodes = {}
        self.parents = {}
        self.children = {}
        self.counter = {}
        self.top_cwe_dict = {
            284: [],
            435: [],
            693: [],
            697: [],
            664: [],
            682: [],
            691: [],
            703: [],
            707: [],
            710: [],
        }

    def add_node(self, cwe_id, name):
        self.nodes[cwe_id] = name
        if cwe_id not in self.parents:
            self.parents[cwe_id] = []
        if cwe_id not in self.children:
            self.children[cwe_id] = []

    def add_vuln(self, cwe_id):

        if cwe_id not in self.counter:
            self.counter[cwe_id] = 0
        self.counter[cwe_id] += 1

        for parent_id in self.children.get(cwe_id, []):
            self.add_vuln(parent_id)

    def add_relationship(self, child_id, parent_id):
        if child_id not in self.children:
            self.children[child_id] = []
        if parent_id not in self.parents:
            self.parents[parent_id] = []

        self.children[child_id].append(parent_id)
        self.parents[parent_id].append(child_id)

    def build_top_cwe(self):
        for cwe_id in self.nodes:
            for top_cwe in self.top_cwe_list(cwe_id):
                self.top_cwe_dict[top_cwe].append(cwe_id)

    def build_from_df(self, df):
        for _, row in df.iterrows():
            cwe_id = row["CWE-ID"]
            name = row["Name"]
            self.add_node(cwe_id, name)

            if pd.notna(row["Related Weaknesses"]):
                relationships = row["Related Weaknesses"].split("::")
                for rel in relationships:

                    if "ChildOf" in rel and "VIEW ID:1000" in rel:
                        parts = rel.split(":")
                        if len(parts) >= 4 and parts[1] == "ChildOf":
                            parent_id = int(parts[3])
                            self.add_relationship(cwe_id, parent_id)
        self.build_top_cwe()

    def get_parents(self, cwe_id):
        return self.parents.get(cwe_id, [])

    def top_cwe(self, cwe_id):
        cwe_list = self.top_cwe_list(cwe_id)
        if 693 in cwe_list:
            return 693
        if 697 in cwe_list:
            return 697
        if 435 in cwe_list:
            return 435
        return cwe_list[0]

    def top_cwe_list(self, cwe_id):

        if cwe_id not in self.children or not self.children[cwe_id]:
            return [cwe_id]

        top_nodes = set()
        for parent_id in self.children[cwe_id]:
            top_nodes.update(self.top_cwe_list(parent_id))

        return list(top_nodes)

    def get_children(self, cwe_id):
        return self.children.get(cwe_id, [])

    def dump_graph(self):
        def _dfs(cwe_id, level=0):

            indent = "    " * level
            node_name = self.nodes.get(cwe_id, "Unknown")
            count = self.counter.get(cwe_id, 0)

            for child_id in self.parents.get(cwe_id, []):
                _dfs(child_id, level + 1)

        root_nodes = []
        for cwe_id in self.nodes:
            if not self.children.get(cwe_id, []):
                root_nodes.append(cwe_id)

        for root in sorted(root_nodes):
            _dfs(root)


cwe_graph = CWEGraph()
cwe_graph.build_from_df(df)
