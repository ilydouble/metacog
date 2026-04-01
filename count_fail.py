def greedy_count(N):
    c25 = N // 25
    rem = N % 25
    c10 = rem // 10
    rem = rem % 10
    c1 = rem
    return c25 + c10 + c1

def min_coins(N):
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

# Count failures
fail_count = 0
for N in range(1, 1001):
    g = greedy_count(N)
    o = min_coins(N)
    if g != o:
        fail_count += 1

print(f"Failures: {fail_count}")
print(f"Successes: {1000 - fail_count}")
