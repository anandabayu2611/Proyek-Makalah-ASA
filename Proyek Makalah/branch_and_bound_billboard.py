# Ananda Putra Bayu
# 24060122140125

import random
import math
import time
import heapq
from dataclasses import dataclass


@dataclass
class Candidate:
    id: int
    x: float
    y: float
    cost: int
    value: int


def generate_candidates(n, seed=42):
    rng = random.Random(seed + n * 13)
    candidates = []

    for i in range(n):
        if i % 5 == 0:
            cx, cy = rng.choice([(250, 250), (750, 250), (250, 750), (750, 750), (500, 500)])
            x = max(0, min(1000, rng.gauss(cx, 120)))
            y = max(0, min(1000, rng.gauss(cy, 120)))
        else:
            x = rng.uniform(0, 1000)
            y = rng.uniform(0, 1000)

        cost = rng.randint(4, 18)
        centrality = 1 - (math.hypot(x - 500, y - 500) / 707.2)
        value = int(rng.randint(80, 230) + 100 * centrality + rng.randint(-20, 60))
        candidates.append(Candidate(i + 1, x, y, cost, value))

    return candidates


def conflict_masks(candidates, min_dist=170):
    n = len(candidates)
    masks = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            distance = math.hypot(candidates[i].x - candidates[j].x, candidates[i].y - candidates[j].y)
            if distance < min_dist:
                masks[i] |= 1 << j
                masks[j] |= 1 << i

    return masks


def branch_and_bound(candidates, budget, min_dist=170):
    n = len(candidates)

    order = sorted(range(n), key=lambda i: candidates[i].value / candidates[i].cost, reverse=True)
    items = [candidates[i] for i in order]

    old_to_new = {old: new for new, old in enumerate(order)}
    original_masks = conflict_masks(candidates, min_dist)
    masks = [0] * n

    for new_i, old_i in enumerate(order):
        new_mask = 0
        for old_j in range(n):
            if original_masks[old_i] & (1 << old_j):
                new_mask |= 1 << old_to_new[old_j]
        masks[new_i] = new_mask

    def upper_bound(index, total_cost, total_value):
        remaining_budget = budget - total_cost
        estimate = total_value
        i = index

        # Bound menggunakan pendekatan fractional knapsack.
        while i < n and remaining_budget > 0:
            if items[i].cost <= remaining_budget:
                remaining_budget -= items[i].cost
                estimate += items[i].value
            else:
                estimate += items[i].value * (remaining_budget / items[i].cost)
                break
            i += 1

        return estimate

    best_value = 0
    best_mask = 0
    nodes = 0
    queue = [(-upper_bound(0, 0, 0), 0, 0, 0, 0)]

    start = time.perf_counter()

    while queue:
        negative_bound, index, total_cost, total_value, selected_mask = heapq.heappop(queue)
        nodes += 1

        if -negative_bound <= best_value:
            continue

        if index == n:
            if total_value > best_value:
                best_value = total_value
                best_mask = selected_mask
            continue

        # Cabang 1: lokasi billboard diambil
        if total_cost + items[index].cost <= budget and not (selected_mask & masks[index]):
            new_cost = total_cost + items[index].cost
            new_value = total_value + items[index].value
            new_mask = selected_mask | (1 << index)

            if new_value > best_value:
                best_value = new_value
                best_mask = new_mask

            bound_value = upper_bound(index + 1, new_cost, new_value)
            if bound_value > best_value:
                heapq.heappush(queue, (-bound_value, index + 1, new_cost, new_value, new_mask))

        # Cabang 2: lokasi billboard tidak diambil
        bound_value = upper_bound(index + 1, total_cost, total_value)
        if bound_value > best_value:
            heapq.heappush(queue, (-bound_value, index + 1, total_cost, total_value, selected_mask))

    elapsed = (time.perf_counter() - start) * 1000

    selected = []
    for new_i, old_i in enumerate(order):
        if best_mask & (1 << new_i):
            selected.append(old_i + 1)

    return best_value, sorted(selected), elapsed, nodes


def main():
    for n in [12, 16, 20, 24, 28]:
        candidates = generate_candidates(n)
        budget = round(0.33 * sum(c.cost for c in candidates))
        best_value, selected, elapsed, nodes = branch_and_bound(candidates, budget)

        print(f"\nJumlah kandidat : {n}")
        print(f"Budget          : {budget}")
        print(f"Nilai terbaik   : {best_value}")
        print(f"Lokasi dipilih  : {selected}")
        print(f"Waktu eksekusi  : {elapsed:.4f} ms")
        print(f"Jumlah node     : {nodes}")


if __name__ == "__main__":
    main()
