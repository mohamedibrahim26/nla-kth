"""
bug_injection.py — AST-based single-statement bug injection for Python functions.

Supported bug types:
  off_by_one  – flip < / <= in comparisons, or perturb range() bound by ±1
  wrong_op    – flip +/- or */÷ in a BinOp
  wrong_var   – swap two Name references inside a BinOp
  wrong_cmp   – flip == / != or > / <

Each injector returns (buggy_source, bug_type, bug_line) or (None, None, None)
if no suitable injection site is found in the given source.
"""

import ast
import copy
import random
from typing import Optional, Tuple

# ── operator flip maps ────────────────────────────────────────────────────────
_CMP_FLIP = {
    ast.Lt:    ast.LtE,
    ast.LtE:   ast.Lt,
    ast.Gt:    ast.GtE,
    ast.GtE:   ast.Gt,
    ast.Eq:    ast.NotEq,
    ast.NotEq: ast.Eq,
}
_ARITH_FLIP = {
    ast.Add:  ast.Sub,
    ast.Sub:  ast.Add,
    ast.Mult: ast.Div,
    ast.Div:  ast.Mult,
}


# ── mutator visitors ──────────────────────────────────────────────────────────

class _OffByOneMutator(ast.NodeTransformer):
    """Flip the first comparison operator found, or perturb a range() bound."""

    def __init__(self):
        self.mutated = False
        self.line: Optional[int] = None

    def visit_Compare(self, node):
        if self.mutated:
            return node
        for i, op in enumerate(node.ops):
            if type(op) in _CMP_FLIP:
                node.ops[i] = _CMP_FLIP[type(op)]()
                self.mutated = True
                self.line = getattr(node, "lineno", None)
                return node
        return self.generic_visit(node)

    def visit_Call(self, node):
        if self.mutated:
            return node
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and len(node.args) == 1
        ):
            delta = random.choice([-1, 1])
            node.args[0] = ast.BinOp(
                left=node.args[0],
                op=ast.Add() if delta > 0 else ast.Sub(),
                right=ast.Constant(value=1),
            )
            ast.fix_missing_locations(node)
            self.mutated = True
            self.line = getattr(node, "lineno", None)
        return self.generic_visit(node)


class _WrongOpMutator(ast.NodeTransformer):
    """Flip the first arithmetic operator found."""

    def __init__(self):
        self.mutated = False
        self.line: Optional[int] = None

    def visit_BinOp(self, node):
        if self.mutated:
            return node
        if type(node.op) in _ARITH_FLIP:
            node.op = _ARITH_FLIP[type(node.op)]()
            self.mutated = True
            self.line = getattr(node, "lineno", None)
        return self.generic_visit(node)


class _WrongVarMutator(ast.NodeTransformer):
    """Swap two Name references inside the first BinOp found."""

    def __init__(self):
        self.mutated = False
        self.line: Optional[int] = None

    def visit_BinOp(self, node):
        if self.mutated:
            return node
        if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
            node.left.id, node.right.id = node.right.id, node.left.id
            self.mutated = True
            self.line = getattr(node, "lineno", None)
        return self.generic_visit(node)


class _WrongCmpMutator(ast.NodeTransformer):
    """Flip == ↔ != or > ↔ < in the first Compare found."""

    def __init__(self):
        self.mutated = False
        self.line: Optional[int] = None

    def visit_Compare(self, node):
        if self.mutated:
            return node
        for i, op in enumerate(node.ops):
            if type(op) in {ast.Eq, ast.NotEq, ast.Gt, ast.Lt}:
                node.ops[i] = _CMP_FLIP[type(op)]()
                self.mutated = True
                self.line = getattr(node, "lineno", None)
                return node
        return self.generic_visit(node)


# ── public API ────────────────────────────────────────────────────────────────

_MUTATORS = {
    "off_by_one": _OffByOneMutator,
    "wrong_op":   _WrongOpMutator,
    "wrong_var":  _WrongVarMutator,
    "wrong_cmp":  _WrongCmpMutator,
}

BUG_TYPES = list(_MUTATORS.keys())


def inject_bug(
    source: str,
    bug_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Attempt to inject a single-statement bug into *source*.

    Parameters
    ----------
    source   : Python source string (ideally a single function definition)
    bug_type : one of BUG_TYPES or None (randomly chosen)
    seed     : RNG seed for reproducibility

    Returns
    -------
    (buggy_source, bug_type_used, bug_line) on success.
    (None, None, None) if no injection site was found or source is invalid.
    """
    if seed is not None:
        random.seed(seed)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None, None, None

    chosen = bug_type or random.choice(BUG_TYPES)
    mutator = _MUTATORS[chosen]()

    new_tree = mutator.visit(copy.deepcopy(tree))

    if not mutator.mutated:
        return None, None, None

    try:
        ast.fix_missing_locations(new_tree)
        buggy_source = ast.unparse(new_tree)
        return buggy_source, chosen, mutator.line
    except Exception:
        return None, None, None


def try_all_bug_types(
    source: str,
    seed: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Try each bug type in random order until one succeeds.
    Returns (buggy_source, bug_type, bug_line) or (None, None, None).
    """
    if seed is not None:
        random.seed(seed)

    order = BUG_TYPES[:]
    random.shuffle(order)
    for bt in order:
        result = inject_bug(source, bug_type=bt, seed=None)
        if result[0] is not None:
            return result
    return None, None, None
