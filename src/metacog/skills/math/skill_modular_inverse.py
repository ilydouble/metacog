"""Skill: Modular Inverse

Standalone skill file — no metacog dependency.
Agent imports directly: from skill_modular_inverse import modular_inverse
"""

SKILL_META = {
    "name": "modular_inverse",
    "description": "Compute modular inverse: find x s.t. a*x ≡ 1 (mod m). Uses Python built-in pow(a,-1,m).",
    "tags": ["number_theory", "modular_arithmetic"],
    "module": "skill_modular_inverse",
    "usage": (
        "from skill_modular_inverse import modular_inverse, solve_linear_congruence\n"
        "x = modular_inverse(3, 7)       # x=5, because 3*5=15≡1 (mod 7)\n"
        "x = solve_linear_congruence(3, 1, 7)  # same: 3x≡1(mod 7) → x=5"
    ),
}


def modular_inverse(a: int, m: int) -> int:
    """Return x such that a*x ≡ 1 (mod m). Raises ValueError if inverse doesn't exist."""
    return pow(a, -1, m)


def solve_linear_congruence(a: int, b: int, m: int) -> int | None:
    """Solve a*x ≡ b (mod m). Returns smallest non-negative x, or None if no solution."""
    from math import gcd
    g = gcd(a, m)
    if b % g != 0:
        return None
    a_, b_, m_ = a // g, b // g, m // g
    return (b_ * pow(a_, -1, m_)) % m_


if __name__ == "__main__":
    print(modular_inverse(3, 7))           # 5
    print(solve_linear_congruence(3, 1, 7))  # 5
    print(solve_linear_congruence(6, 4, 10))  # 4  (6*4=24≡4 mod10)
    print(solve_linear_congruence(2, 1, 4))   # None (gcd(2,4)=2, 1%2≠0)

