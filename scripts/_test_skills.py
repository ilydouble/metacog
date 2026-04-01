#!/usr/bin/env python3
"""验证 4 个 seed skills 和 registry 扫描"""
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
SKILLS_DIR = _root / "src" / "metacog" / "skills" / "math"
sys.path.insert(0, str(SKILLS_DIR))
sys.path.insert(0, str(_root / "src" / "mini-swe-agent" / "src"))
sys.path.insert(0, str(_root / "src"))

ok = True

# ── 1. prime_utils ──────────────────────────────────────────────────────────
print("=== 1. skill_prime_utils ===")
from skill_prime_utils import factorize, primes_up_to, is_prime, euler_phi, divisors, num_divisors
assert factorize(360) == {2: 3, 3: 2, 5: 1}, factorize(360)
assert primes_up_to(30) == [2,3,5,7,11,13,17,19,23,29]
assert is_prime(97) and not is_prime(100)
assert euler_phi(12) == 4
assert divisors(12) == [1,2,3,4,6,12]
assert num_divisors(360) == 24
print("  ✅ factorize, primes_up_to, is_prime, euler_phi, divisors, num_divisors")

# ── 2. crt ──────────────────────────────────────────────────────────────────
print("=== 2. skill_crt ===")
from skill_crt import crt, crt_list
assert crt(2, 3, 3, 5) == (8, 15), crt(2, 3, 3, 5)
assert crt_list([2, 3, 2], [3, 5, 7]) == (23, 105), crt_list([2,3,2],[3,5,7])
assert crt(0, 4, 1, 2) == (None, None)
print("  ✅ crt, crt_list")

# ── 3. combinatorics ────────────────────────────────────────────────────────
print("=== 3. skill_combinatorics ===")
from skill_combinatorics import C, P, catalan, derangement, stirling2, multinomial, C_mod
assert C(10, 3) == 120
assert P(10, 3) == 720
assert catalan(5) == 42
assert derangement(4) == 9
assert stirling2(4, 2) == 7
assert multinomial(4, [2, 1, 1]) == 12
val = C_mod(100, 50, 10**9 + 7)
assert isinstance(val, int) and val > 0
print(f"  ✅ C, P, catalan, derangement, stirling2, multinomial, C_mod={val}")

# ── 4. sympy_solve ──────────────────────────────────────────────────────────
print("=== 4. skill_sympy_solve ===")
from skill_sympy_solve import solve_eq, solve_system, simplify_expr, poly_roots, sum_series, factor_expr
from sympy import Symbol

roots = solve_eq("x**2 - 5*x + 6", "x")
assert set(roots) == {2, 3}, roots

sol = solve_system(["x+y-5", "x-y-1"], ["x", "y"])
x, y = Symbol("x"), Symbol("y")
assert sol[str(x)] == 3 and sol[str(y)] == 2, sol

s = simplify_expr("(x**2-1)/(x-1)")
assert "x" in s and "1" in s, s

pr = poly_roots("x**3 - 6*x**2 + 11*x - 6")
assert set(pr) == {1, 2, 3}, pr

sm = sum_series("k**2", "k", 1, "n")
assert "n" in sm, sm

fe = factor_expr("x**3 - x")
assert "x" in fe, fe

print(f"  ✅ solve_eq={roots}, solve_system={sol}")
print(f"  ✅ simplify={s}, poly_roots={pr}")
print(f"  ✅ sum_series={sm}, factor={fe}")

# ── 5. SkillRegistry 扫描 ───────────────────────────────────────────────────
print("=== 5. SkillRegistry 扫描 ===")
from metacog.skills.registry import SkillRegistry
registry = SkillRegistry()
loaded = registry.register_from_dir(SKILLS_DIR)
names = [s.name for s in loaded]
print(f"  加载 {len(loaded)} 个 skills: {names}")
assert "modular_inverse" in names
assert "prime_utils" in names
assert "crt" in names
assert "combinatorics" in names
assert "sympy_solve" in names
print("  ✅ 全部注册成功")
print()
print(registry.as_prompt_text())

print("\n🎉 所有测试通过！")

