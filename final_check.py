import math

def count_beautiful(b):
    """Count b-eautiful integers for base b."""
    count = 0
    solutions = []
    for d1 in range(1, b):
        for d0 in range(b):
            n = d1 * b + d0
            s = d1 + d0
            if s * s == n:
                count += 1
                solutions.append((d1, d0, n))
    return count, solutions

# Find the least b >= 2 such that there are more than 10 b-eautiful integers
b = 2
while True:
    c, sols = count_beautiful(b)
    if c > 10:
        print(f"First base with count > 10 is b={b} with count={c}")
        print(f"Solutions: {sols}")
        break
    b += 1
