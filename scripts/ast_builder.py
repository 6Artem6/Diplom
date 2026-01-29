import importlib
import inspect
import pkgutil
import ast
from pathlib import Path
import sys

project_root = Path("/mnt/c/Users/Bobsky/PycharmProjects/Diplom/apps/sitegpt")

for py_file in project_root.rglob("*.py"):
    with open(py_file, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read())
        except Exception as e:
            print(f"⚠️ Cannot import {py_file}: {e}")
            continue

    for cls in [n for n in node.body if isinstance(n, ast.ClassDef)]:
        print(f"Class: {cls.name} ({py_file.name.split('/')[-1]})")
        for func in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            args = []
            for a in func.args.args:
                if a.annotation:
                    arg_type = ast.unparse(a.annotation)
                    args.append(f"{a.arg}: {arg_type}")
                else:
                    args.append(a.arg)
            returns = ast.unparse(func.returns) if func.returns else "None"
            print(f"  Method: {func.name}({', '.join(args)}) -> {returns}")
        print()
