"""Skill: SymPy Solver Utilities

Standalone skill file — no metacog dependency.
Agent imports: from skill_sympy_solve import solve_eq, solve_system, simplify_expr, ...
"""

SKILL_META = {
    "name": "sympy_solve",
    "description": (
        "SymPy wrappers for equation solving, simplification, summation, "
        "polynomial roots, and inequality solving."
    ),
    "tags": ["algebra", "sympy", "equations"],
    "module": "skill_sympy_solve",
    "usage": (
        "from skill_sympy_solve import solve_eq, solve_system, simplify_expr, poly_roots, sum_series\n"
        "solve_eq('x**2 - 5*x + 6', 'x')          # [2, 3]\n"
        "solve_system(['x+y-5', 'x-y-1'], ['x','y'])  # {x:3, y:2}\n"
        "simplify_expr('(x**2-1)/(x-1)')           # x+1\n"
        "poly_roots('x**3 - 6*x**2 + 11*x - 6')   # [1,2,3]\n"
        "sum_series('k**2', 'k', 1, 'n')           # n*(n+1)*(2*n+1)/6"
    ),
}


def solve_eq(expr_str: str, var: str = "x") -> list:
    """Solve expr == 0 for var. Returns list of solutions (may be symbolic).

    Example:
        solve_eq('x**2 - 5*x + 6', 'x')  # [2, 3]
    """
    from sympy import symbols, solve, sympify
    x = symbols(var)
    expr = sympify(expr_str)
    return solve(expr, x)


def solve_system(exprs: list[str], vars: list[str]) -> dict:
    """Solve a system of equations (each expr == 0).

    Example:
        solve_system(['x+y-5', 'x-y-1'], ['x','y'])  # {x:3, y:2}
    """
    from sympy import symbols, solve, sympify
    sym_vars = symbols(" ".join(vars))
    if len(vars) == 1:
        sym_vars = (sym_vars,)
    parsed = [sympify(e) for e in exprs]
    result = solve(parsed, sym_vars)
    # Normalize output to dict with string keys
    if isinstance(result, dict):
        return {str(k): v for k, v in result.items()}
    if isinstance(result, list) and result and isinstance(result[0], tuple):
        return {str(v): result[0][i] for i, v in enumerate(sym_vars)}
    return result


def simplify_expr(expr_str: str) -> str:
    """Simplify a symbolic expression and return as string.

    Example:
        simplify_expr('(x**2-1)/(x-1)')  # 'x + 1'
    """
    from sympy import sympify, simplify
    return str(simplify(sympify(expr_str)))


def poly_roots(poly_str: str, var: str = "x") -> list:
    """Find all roots of a polynomial (integer/rational roots first, then symbolic).

    Example:
        poly_roots('x**3 - 6*x**2 + 11*x - 6')  # [1, 2, 3]
    """
    from sympy import symbols, Poly, roots, sympify
    x = symbols(var)
    p = Poly(sympify(poly_str), x)
    root_dict = roots(p, x)
    result = []
    for r, mult in root_dict.items():
        result.extend([r] * mult)
    return sorted(result, key=lambda r: (r.is_real, float(r)) if r.is_real else (False, 0))


def sum_series(term_str: str, var: str, start, end) -> str:
    """Compute symbolic sum of term from var=start to var=end.

    Example:
        sum_series('k**2', 'k', 1, 'n')  # n*(n+1)*(2*n+1)/6
    """
    from sympy import symbols, summation, sympify, simplify
    k = symbols(var)
    n_sym = sympify(end)
    term = sympify(term_str)
    result = summation(term, (k, sympify(start), n_sym))
    return str(simplify(result))


def factor_expr(expr_str: str) -> str:
    """Factor a polynomial expression.

    Example:
        factor_expr('x**3 - x')  # 'x*(x - 1)*(x + 1)'
    """
    from sympy import sympify, factor
    return str(factor(sympify(expr_str)))


def expand_expr(expr_str: str) -> str:
    """Expand a polynomial expression.

    Example:
        expand_expr('(x+1)**3')  # 'x**3 + 3*x**2 + 3*x + 1'
    """
    from sympy import sympify, expand
    return str(expand(sympify(expr_str)))


if __name__ == "__main__":
    print(solve_eq("x**2 - 5*x + 6", "x"))           # [2, 3]
    print(solve_system(["x+y-5", "x-y-1"], ["x", "y"]))  # {x:3, y:2}
    print(simplify_expr("(x**2-1)/(x-1)"))            # x + 1
    print(poly_roots("x**3 - 6*x**2 + 11*x - 6"))    # [1, 2, 3]
    print(sum_series("k**2", "k", 1, "n"))            # n*(n+1)*(2*n+1)/6
    print(factor_expr("x**3 - x"))                    # x*(x-1)*(x+1)
    print(expand_expr("(x+1)**3"))                    # x**3+3*x**2+3*x+1

