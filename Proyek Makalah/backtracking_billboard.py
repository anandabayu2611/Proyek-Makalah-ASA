# Ananda Putra Bayu
# 24060122140125

import random
import math
import time
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


def backtracking(candidates, budget, min_dist=170):
    n = len(candidates)
    masks = conflict_masks(candidates, min_dist)

    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + candidates[i].value

    best_value = 0
    best_mask = 0
    nodes = 0

    def dfs(index, total_cost, total_value, selected_mask):
        nonlocal best_value, best_mask, nodes
        nodes += 1

        if total_value + suffix[index] <= best_value:
            return

        if index == n:
            if total_value > best_value:
                best_value = total_value
                best_mask = selected_mask
            return

        # Cabang 1: lokasi billboard diambil
        if total_cost + candidates[index].cost <= budget and not (selected_mask & masks[index]):
            dfs(
                index + 1,
                total_cost + candidates[index].cost,
                total_value + candidates[index].value,
                selected_mask | (1 << index)
            )

        # Cabang 2: lokasi billboard tidak diambil
        dfs(index + 1, total_cost, total_value, selected_mask)

    start = time.perf_counter()
    dfs(0, 0, 0, 0)
    elapsed = (time.perf_counter() - start) * 1000

    selected = [i + 1 for i in range(n) if best_mask & (1 << i)]
    return best_value, selected, elapsed, nodes


def main():
    for n in [12, 16, 20, 24, 28]:
        candidates = generate_candidates(n)
        budget = round(0.33 * sum(c.cost for c in candidates))
        best_value, selected, elapsed, nodes = backtracking(candidates, budget)

        print(f"\nJumlah kandidat : {n}")
        print(f"Budget          : {budget}")
        print(f"Nilai terbaik   : {best_value}")
        print(f"Lokasi dipilih  : {selected}")
        print(f"Waktu eksekusi  : {elapsed:.4f} ms")
        print(f"Jumlah node     : {nodes}")


if __name__ == "__main__":
    main()
