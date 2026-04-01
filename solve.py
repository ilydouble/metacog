# Problem: Find number of N in [1, 1000] where greedy algorithm succeeds for coins {1, 10, 25}
# Greedy succeeds if no other combination uses strictly fewer coins.

def greedy_count(N):
    # Greedy algorithm: take as many 25s as possible, then 10s, then 1s
    c25 = N // 25
    rem = N % 25
    c10 = rem // 10
    rem = rem % 10
    c1 = rem
    return c25 + c10 + c1

def min_coins(N):
    # Find minimum number of coins to make N using dynamic programming or brute force
    # Since we have coins 1, 10, 25, we can iterate over possible counts of 25 and 10
    min_c = float('inf')
    for c25 in range(N // 25 + 1):
        rem_after_25 = N - 25 * c25
        for c10 in range(rem_after_25 // 10 + 1):
            rem_after_10 = rem_after_25 - 10 * c10
            c1 = rem_after_10
            total = c25 + c10 + c1
            if total < min_c:
                min_c = total
    return min_c

# Check for N from 1 to 1000
count = 0
for N in range(1, 1001):
    greedy = greedy_count(N)
    optimal = min_coins(N)
    if greedy == optimal:
        count += 1

print(count)
