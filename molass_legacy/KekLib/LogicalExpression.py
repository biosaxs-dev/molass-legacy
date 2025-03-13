# coding: utf-8
"""
    LogicalExpression.py

    Copyright (c) 2021, Masatsuyo Takahashi, KEK-PF
"""

import sys
import re
import inspect
import ast

expr_re = re.compile(r"return\s+(.+)?\s+$")

class PrintNodeVisitor(ast.NodeVisitor):
    def visit(self, node):
        print(node)
        return super().visit(node)

class LogicalExpression:
    def __init__(self, closure):
        self.closure = closure
        self.structure = self.analyze(closure)

    def analyze(self, closure):
        src = inspect.getsource(closure)
        print(src)
        m = expr_re.search(src)
        if m:
            expr_src = m.group(1)
        else:
            assert False

        print("expr_src=", expr_src)
        expr = ast.parse(expr_src, mode='eval')
        print(ast.dump(expr, indent=4))

        visitor = PrintNodeVisitor()
        visitor.visit(expr)

    def tracer(self, *args):
        print("---- tracer:", *args)

    def evaluate(self):
        sys.settrace(self.tracer)
        ret = self.closure()
        sys.settrace(None)
        why = self.get_reason(ret)
        return ret, why

    def get_reason(self, judge):
        return None

def spike():
    pass
