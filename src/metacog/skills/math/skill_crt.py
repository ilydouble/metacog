"""Skill: Chinese Remainder Theorem (CRT)

Standalone skill file — no metacog dependency.
Agent imports: from skill_crt import crt, crt_list
"""

SKILL_META = {
    "name": "crt",
    "description": (
        "Chinese Remainder Theorem solver. "
        "Find x s.t. x ≡ r_i (mod m_i) for all i. "
        "Works for both coprime and non-coprime moduli (via extended CRT)."
    ),
    "tags": ["number_theory", "modular_arithmetic", "crt"],
    "module": "skill_crt",
    "usage": (
        "from skill_crt import crt, crt_list\n"
        "# x ≡ 2 (mod 3) and x ≡ 3 (mod 5) → x=8 (mod 15)\n"
        "x, m = crt(2, 3, 3, 5)     # x=8, m=15\n"
        "# Multiple congruences at once\n"
        "x, m = crt_list([2, 3, 2], [3, 5, 7])  # x=23, m=105"
    ),
}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) s.t. a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = _extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def crt(r1: int, m1: int, r2: int, m2: int) -> tuple[int, int] | tuple[None, None]:
    """Solve x ≡ r1 (mod m1) and x ≡ r2 (mod m2).

    Returns (x, lcm) where x is the smallest non-negative solution,
    or (None, None) if no solution exists.
    """
    g, p, _ = _extended_gcd(m1, m2)
    if (r2 - r1) % g != 0:
        return None, None   # no solution
    lcm = m1 * m2 // g
    diff = (r2 - r1) // g
    x = (r1 + m1 * (diff * p % (m2 // g))) % lcm
    return x, lcm


def crt_list(
    remainders: list[int], moduli: list[int]
) -> tuple[int, int] | tuple[None, None]:
    """Solve system x ≡ r_i (mod m_i) for all i.

    Returns (x, M) where M = lcm of all moduli,
    or (None, None) if the system is inconsistent.
    """
    if len(remainders) != len(moduli):
        raise ValueError("remainders and moduli must have the same length")
    x, m = remainders[0], moduli[0]
    for r, mod in zip(remainders[1:], moduli[1:]):
        x, m = crt(x, m, r, mod)
        if x is None:
            return None, None
    return x % m, m


if __name__ == "__main__":
    # x ≡ 2 (mod 3), x ≡ 3 (mod 5)  →  x=8, m=15
    print(crt(2, 3, 3, 5))           # (8, 15)
    # x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)  →  x=23, m=105
    print(crt_list([2, 3, 2], [3, 5, 7]))  # (23, 105)
    # No solution: x ≡ 0 (mod 4), x ≡ 1 (mod 2)
    print(crt(0, 4, 1, 2))           # (None, None)

