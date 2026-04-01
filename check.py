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

# Check bases around 210-215 more carefully
for b in range(205, 220):
    c, sols = count_beautiful(b)
    print(f"b={b}, count={c}")
