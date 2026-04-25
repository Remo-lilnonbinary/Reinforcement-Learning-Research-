import os, numpy as np, pandas as pd

rng = np.random.default_rng(42)

RACES = [
    ("IA_3_2018", 47, 0.487),
    ("VA_7_2018", 52, 0.491),
    ("AZ_1_2020", 48, 0.490),
    ("IA_2_2020", 50, 0.499),
    ("CO_3_2022", 45, 0.485),
    ("PA_8_2022", 49, 0.490),
    ("MI_10_2022", 51, 0.488),
]


def make_race(race_id, n, mean_lean):
    rows = []
    for i in range(n):
        lean = float(np.clip(rng.normal(mean_lean, 0.10), 0.20, 0.80))
        mixed = abs(lean - 0.5) < 0.10
        swing = 0.18 if mixed else 0.08
        hard_opp = (1 - lean) * 0.55
        soft_opp = (1 - lean) * 0.20
        soft_sup = lean * 0.20
        hard_sup_high = lean * 0.35
        hard_sup_low = lean * 0.20
        total = hard_opp + soft_opp + swing + soft_sup + hard_sup_high + hard_sup_low
        rows.append({
            "precinct_id": f"P_{i:03d}",
            "race_id": race_id,
            "n_voters": int(rng.lognormal(7.5, 0.4)),
            "density": str(rng.choice(["urban", "suburban", "rural"], p=[0.3, 0.5, 0.2])),
            "baseline_lean": round(lean, 3),
            "hard_opp": round(hard_opp / total, 4),
            "soft_opp": round(soft_opp / total, 4),
            "swing": round(swing / total, 4),
            "soft_sup": round(soft_sup / total, 4),
            "hard_sup_high": round(hard_sup_high / total, 4),
            "hard_sup_low": round(hard_sup_low / total, 4),
        })
    return pd.DataFrame(rows)


os.makedirs("data", exist_ok=True)
for race_id, n, lean in RACES:
    df = make_race(race_id, n, lean)
    df.to_parquet(f"data/{race_id}.parquet")
    print(race_id, len(df), round(df.baseline_lean.mean(), 3))