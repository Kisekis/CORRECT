from dataclasses import dataclass
from typing import List, Tuple, Dict, NamedTuple

CalleeMethod = NamedTuple(
    "CalleeMethod",
    [("filename", str), ("method_name", str), ("raw_code", str), ("depth", int)],
)
TypeDef = NamedTuple("TypeDef", [("type_def", str), ("name", str)])
VulnMethod = NamedTuple(
    "VulnMethod",
    [("filename", str), ("method_name", str), ("raw_code", str), ("start_line", int)],
)


@dataclass
class VulnObj:
    code: str
    cwe: str
    is_vulnerable: bool


@dataclass
class Example:
    code: str
    cwe: str
    is_vulnerable: bool
    explanation: str


@dataclass
class VulnPair:
    cwe: List[str]
    vuln: str
    patched: str
    name: str
    method: List[str]
    context: str


@dataclass
class VulnContext:
    calleeMethods: List[CalleeMethod]
    typeDefs: List[TypeDef]
    globalVars: List[str]
    importContext: List[str]
    visitedLines_before: Dict[str, List[int]]
    visitedLines_after: Dict[str, List[int]]
    visitedParams: Dict[str, List[str]]

    def __str__(self) -> str:
        parts = []

        if self.calleeMethods:
            parts.append("Callee Methods:")
            for method in self.calleeMethods:
                parts.append(
                    f"  - {method.filename}::{method.method_name}::{method.raw_code} (深度: {method.depth})"
                )

        if self.typeDefs:
            parts.append("Type Definitions:")
            for type_def in self.typeDefs:
                parts.append(f"  - {type_def.name}: {type_def.type_def}")

        if self.globalVars:
            parts.append("Global Variables:")
            for var in self.globalVars:
                parts.append(f"  - {var}")

        if self.importContext:
            parts.append("Import Statements:")
            for imp in self.importContext:
                parts.append(f"  - {imp}")

        if self.visitedLines_before:
            parts.append("Visited Lines (before):")
            for filename, lines in self.visitedLines_before.items():
                parts.append(f"  - {filename}: {', '.join(map(str, lines))}")

        if self.visitedLines_after:
            parts.append("Visited Lines (after):")
            for filename, lines in self.visitedLines_after.items():
                parts.append(f"  - {filename}: {', '.join(map(str, lines))}")

        if self.visitedParams:
            parts.append("Visited Parameters:")
            for method, params in self.visitedParams.items():
                parts.append(f"  - {method}: {', '.join(params)}")

        return "\n".join(parts)


@dataclass
class VulnPairWithContext:
    cwe: List[str]
    vuln: List[VulnMethod]
    patched: List[VulnMethod]
    name: str
    context: VulnContext
