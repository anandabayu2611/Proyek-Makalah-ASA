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


def genetic_algorithm(candidates, budget, min_dist=170, pop_size=20, generations=25, seed=123):
    rng = random.Random(seed + len(candidates))
    n = len(candidates)
    masks = conflict_masks(candidates, min_dist)
    density = [c.value / c.cost for c in candidates]

    def repair(bits):
        bits = bits[:]

        # Memperbaiki konflik jarak.
        changed = True
        while changed:
            changed = False

            for i in range(n):
                if not bits[i]:
                    continue

                for j in range(i + 1, n):
                    if bits[j] and (masks[i] & (1 << j)):
                        drop = i if density[i] < density[j] else j
                        bits[drop] = 0
                        changed = True
                        break

                if changed:
                    break

        # Memperbaiki solusi yang melebihi anggaran.
        while sum(candidates[i].cost for i in range(n) if bits[i]) > budget:
            chosen = [i for i in range(n) if bits[i]]
            drop = min(chosen, key=lambda i: density[i])
            bits[drop] = 0

        return bits

    def fitness(bits):
        fixed_bits = repair(bits)
        total_value = sum(candidates[i].value for i in range(n) if fixed_bits[i])
        return total_value, fixed_bits

    population = []
    for _ in range(pop_size):
        chromosome = [1 if rng.random() < 0.35 else 0 for _ in range(n)]
        value, chromosome = fitness(chromosome)
        population.append((value, chromosome))

    best = max(population, key=lambda x: x[0])
    start = time.perf_counter()
    evaluations = pop_size

    for _ in range(generations):
        new_population = []

        # Elitism: empat individu terbaik dipertahankan.
        new_population.extend(sorted(population, key=lambda x: x[0], reverse=True)[:4])

        while len(new_population) < pop_size:
            def tournament_selection():
                return max(rng.sample(population, 3), key=lambda x: x[0])[1]

            parent1 = tournament_selection()
            parent2 = tournament_selection()

            # Crossover uniform
            child = []
            for gene1, gene2 in zip(parent1, parent2):
                child.append(gene1 if rng.random() < 0.5 else gene2)

            # Mutasi
            for i in range(n):
                if rng.random() < 1 / n:
                    child[i] = 1 - child[i]

            value, child = fitness(child)
            evaluations += 1
            new_population.append((value, child))

        population = new_population
        current_best = max(population, key=lambda x: x[0])

        if current_best[0] > best[0]:
            best = current_best

    elapsed = (time.perf_counter() - start) * 1000
    selected = [i + 1 for i, bit in enumerate(best[1]) if bit]

    return best[0], selected, elapsed, evaluations


def main():
    for n in [12, 16, 20, 24, 28]:
        candidates = generate_candidates(n)
        budget = round(0.33 * sum(c.cost for c in candidates))

        best_values = []
        best_selected = []
        times = []

        # GA dijalankan 5 kali karena hasilnya dipengaruhi proses acak.
        for seed in range(5):
            value, selected, elapsed, evaluations = genetic_algorithm(candidates, budget, seed=1000 + seed)
            best_values.append(value)
            best_selected.append(selected)
            times.append(elapsed)

        best_index = best_values.index(max(best_values))

        print(f"\nJumlah kandidat : {n}")
        print(f"Budget          : {budget}")
        print(f"Nilai terbaik   : {best_values[best_index]}")
        print(f"Rata-rata nilai : {sum(best_values) / len(best_values):.2f}")
        print(f"Lokasi dipilih  : {best_selected[best_index]}")
        print(f"Rata-rata waktu : {sum(times) / len(times):.4f} ms")


if __name__ == "__main__":
    main()
