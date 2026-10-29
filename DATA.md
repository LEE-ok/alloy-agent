# Data

The only training data is **Steel Strength Data** (312 steels: 13 alloying elements in wt% →
yield strength, ultimate tensile strength, elongation).

| | |
|---|---|
| Source | Hacking Materials, Figshare article [7250453](https://figshare.com/articles/Steel_Strength_Data/7250453) |
| License | MIT (as stated on the Figshare record) |
| Original source | Citrine Informatics, "Mechanical properties of some steels", citrination.com/datasets/153092 (service retired; upstream license could not be verified) |
| Benchmark | Dunn et al., "Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm", npj Comput. Mater. 6, 138 (2020) |
| File | `steel_strength.csv`, SHA256 `53c995e14371098ff8f6186e43c2a6d833650d352dc0d77118751cf6b06727d7` |

The CSV is **not committed**. `alloy_agent.data.load()` downloads it on first use into `data/`
and verifies the checksum.

Scope note: the data is mostly maraging / ultra-high-strength steels (yield 1006–2510 MPa).
Anything outside the training ranges is flagged as out of domain by `predict`/`search`/`pareto`, not hidden.
