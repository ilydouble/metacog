"""Skill: Combinatorics

Standalone skill file — no metacog dependency.
Agent imports: from skill_combinatorics import C, P, catalan, stirling2, ...
"""

SKILL_META = {
    "name": "combinatorics",
    "description": (
        "Combinatorics utilities: C(n,k), P(n,k), Catalan numbers, "
        "Stirling numbers (2nd kind), derangements, multinomials, "
        "and modular binomial coefficient."
    ),
    "tags": ["combinatorics", "counting"],
    "module": "skill_combinatorics",
    "usage": (
        "from skill_combinatorics import C, P, catalan, derangement, C_mod, multinomial\n"
        "C(10, 3)              # 120\n"
        "P(10, 3)              # 720\n"
        "catalan(5)            # 42\n"
        "derangement(4)        # 9\n"
        "C_mod(100, 50, 10**9+7)   # C(100,50) mod (10^9+7)\n"
        "multinomial(4, [2,1,1])   # 4!/(2!1!1!) = 12"
    ),
}

from math import factorial, gcd


def C(n: int, k: int) -> int:
    """Binomial coefficient C(n, k) = n! / (k! * (n-k)!)."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // (factorial(k) * factorial(n - k))


def P(n: int, k: int) -> int:
    """Permutations P(n, k) = n! / (n-k)!."""
    if k < 0 or k > n:
        return 0
    return factorial(n) // factorial(n - k)


def catalan(n: int) -> int:
    """n-th Catalan number: C(2n, n) / (n+1)."""
    return C(2 * n, n) // (n + 1)


def derangement(n: int) -> int:
    """Number of derangements of n elements (subfactorial !n)."""
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


def stirling2(n: int, k: int) -> int:
    """Stirling number of the second kind S(n, k):
    number of ways to partition n elements into k non-empty subsets.
    """
    if n == k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    # Use inclusion-exclusion: S(n,k) = (1/k!) * sum_{j=0}^{k} (-1)^j * C(k,j) * (k-j)^n
    result = sum(
        ((-1) ** j) * C(k, j) * (k - j) ** n for j in range(k + 1)
    )
    return result // factorial(k)


def multinomial(n: int, groups: list[int]) -> int:
    """Multinomial coefficient n! / (k1! * k2! * ... * km!) where sum(groups) == n."""
    if sum(groups) != n:
        raise ValueError(f"sum(groups)={sum(groups)} != n={n}")
    result = factorial(n)
    for k in groups:
        result //= factorial(k)
    return result


def _modinv(a: int, m: int) -> int:
    return pow(a, -1, m)


def C_mod(n: int, k: int, mod: int) -> int:
    """C(n, k) mod p using Lucas' theorem for prime p, or direct computation."""
    if k < 0 or k > n:
        return 0
    # Direct computation via numerator/denominator mod mod
    num = 1
    den = 1
    for i in range(k):
        num = num * (n - i) % mod
        den = den * (i + 1) % mod
    return num * _modinv(den, mod) % mod


if __name__ == "__main__":
    print(C(10, 3))               # 120
    print(P(10, 3))               # 720
    print(catalan(5))             # 42
    print(derangement(4))         # 9
    print(stirling2(4, 2))        # 7
    print(multinomial(4, [2, 1, 1]))  # 12
    print(C_mod(100, 50, 10**9 + 7))  # C(100,50) mod 1e9+7

