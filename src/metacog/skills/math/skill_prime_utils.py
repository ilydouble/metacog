"""Skill: Prime Utilities

Standalone skill file — no metacog dependency.
Agent imports: from skill_prime_utils import factorize, primes_up_to, is_prime, euler_phi
"""

SKILL_META = {
    "name": "prime_utils",
    "description": (
        "Prime factorization, sieve of Eratosthenes, primality test, "
        "Euler's totient φ(n), and divisor utilities."
    ),
    "tags": ["number_theory", "primes"],
    "module": "skill_prime_utils",
    "usage": (
        "from skill_prime_utils import factorize, primes_up_to, is_prime, euler_phi, divisors\n"
        "factorize(360)          # {2:3, 3:2, 5:1}\n"
        "primes_up_to(30)        # [2,3,5,7,11,13,17,19,23,29]\n"
        "is_prime(97)            # True\n"
        "euler_phi(12)           # 4\n"
        "divisors(12)            # [1,2,3,4,6,12]"
    ),
}


def factorize(n: int) -> dict[int, int]:
    """Return prime factorization of n as {prime: exponent}."""
    if n <= 1:
        return {}
    factors: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def primes_up_to(limit: int) -> list[int]:
    """Sieve of Eratosthenes: return all primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
    return [i for i, v in enumerate(sieve) if v]


def is_prime(n: int) -> bool:
    """Deterministic primality test (Miller-Rabin for n < 3,215,031,751)."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Write n-1 as 2^r * d
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def euler_phi(n: int) -> int:
    """Euler's totient φ(n): count integers in [1,n] coprime to n."""
    result = n
    for p in factorize(n):
        result -= result // p
    return result


def divisors(n: int) -> list[int]:
    """Return sorted list of all positive divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def num_divisors(n: int) -> int:
    """Return the number of positive divisors of n (using factorization)."""
    result = 1
    for exp in factorize(n).values():
        result *= exp + 1
    return result


if __name__ == "__main__":
    print(factorize(360))        # {2:3, 3:2, 5:1}
    print(primes_up_to(30))      # [2,3,5,7,11,13,17,19,23,29]
    print(is_prime(97))          # True
    print(euler_phi(12))         # 4
    print(divisors(12))          # [1,2,3,4,6,12]
    print(num_divisors(360))     # 24

