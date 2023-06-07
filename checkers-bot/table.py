from pathlib import Path

from loader import load


path = Path.cwd() / "data" / "q_table_10000000epi_0.5a_0.95g.json"
file = load(path)

many_occurances = [v for v in file.values() if v not in (0, 0.5, -0.5)]
num_many_occurances = len(many_occurances)

print()
print("Q-table statistics:")
print(f"Total: {len(file)}")
print(f"Many occurances: {num_many_occurances}")
print()