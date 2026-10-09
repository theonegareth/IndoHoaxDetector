"""Dataset statistics + duplicate/leakage audit (reviewer Q1, Q4, Q5, Q17)."""
import os, re, json
import pandas as pd, numpy as np

S = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(S, "preproc_with_meta.csv"))
raw = pd.read_csv(os.path.join(S, "combined_datasets_clean.csv"))
df["text_clean"] = df["text_clean"].astype(str)

SRC_META = {
    "cnn_cleaned":              ("CNN Indonesia",        "news",       "Deteksi Berita Hoaks Indo"),
    "detik_cleaned":            ("Detik.com",            "news",       "Deteksi Berita Hoaks Indo"),
    "kompas_cleaned":           ("Kompas.com",           "news",       "Deteksi Berita Hoaks Indo"),
    "turnbackhoax_cleaned":     ("TurnBackHoax.id",      "fact-check", "Deteksi Berita Hoaks Indo"),
    "cnn_10k_cleaned":          ("CNN Indonesia",        "news",       "Indonesian Fact and Hoax Political News"),
    "kompas_4k_cleaned":        ("Kompas.com",           "news",       "Indonesian Fact and Hoax Political News"),
    "tempo_6k_cleaned":         ("Tempo.co",             "news",       "Indonesian Fact and Hoax Political News"),
    "turnbackhoax_10k_cleaned": ("TurnBackHoax.id",      "fact-check", "Indonesian Fact and Hoax Political News"),
    "komdigi_hoaks":            ("Komdigi (Kominfo)",    "government", "Indonesian Hoax News Dataset"),
}

out = {}

# ---- Q1 / Q17: composition by source -------------------------------------
rows = []
for s, g in df.groupby("source"):
    outlet, genre, origin = SRC_META.get(s, (s, "?", "?"))
    toks = g["text_clean"].str.split().str.len()
    rows.append({
        "source_key": s, "outlet": outlet, "genre": genre, "origin_dataset": origin,
        "n": len(g),
        "pct": 100 * len(g) / len(df),
        "label": "hoax" if g["label_encoded"].mean() > 0.5 else "factual",
        "n_factual": int((g["label_encoded"] == 0).sum()),
        "n_hoax": int((g["label_encoded"] == 1).sum()),
        "mean_tokens": toks.mean(), "median_tokens": toks.median(), "max_tokens": toks.max(),
        "mean_raw_chars": g["raw_len"].mean(), "max_raw_chars": g["raw_len"].max(),
    })
comp = pd.DataFrame(rows).sort_values("n", ascending=False)
comp.to_csv(os.path.join(S, "out_source_composition.csv"), index=False)
print("=== SOURCE COMPOSITION (Q1, Q17) ===")
print(comp[["source_key","outlet","genre","n","pct","label","mean_tokens","median_tokens"]].to_string(index=False))

# ---- the confound ---------------------------------------------------------
ct = pd.crosstab(df["source"], df["label_encoded"])
print("\n=== SOURCE x LABEL (the confound) ===")
print(ct)
pure = (ct.min(axis=1) == 0).all()
print(f"\nEvery source is single-label: {pure}")

# ---- Q17: overall stats ---------------------------------------------------
tok = df["text_clean"].str.split().str.len()
vocab = set()
for t in df["text_clean"]:
    vocab.update(t.split())
print("\n=== CORPUS STATISTICS (Q17) ===")
stats = {
    "n_documents": len(df),
    "n_factual": int((df["label_encoded"] == 0).sum()),
    "n_hoax": int((df["label_encoded"] == 1).sum()),
    "pct_factual": round(100*(df["label_encoded"] == 0).mean(), 2),
    "pct_hoax": round(100*(df["label_encoded"] == 1).mean(), 2),
    "n_sources": df["source"].nunique(),
    "n_distinct_outlets": len({SRC_META.get(s, (s,))[0] for s in df["source"].unique()}),
    "vocab_size_after_preprocessing": len(vocab),
    "mean_tokens": round(tok.mean(), 1), "median_tokens": float(tok.median()),
    "p95_tokens": float(tok.quantile(.95)), "max_tokens": int(tok.max()), "min_tokens": int(tok.min()),
    "mean_raw_chars": round(df["raw_len"].mean(), 1), "max_raw_chars": int(df["raw_len"].max()),
}
for k, v in stats.items(): print(f"  {k:34} {v}")

print("\n=== TOKEN LENGTH BY CLASS ===")
print(df.groupby("label_encoded")["text_clean"].apply(lambda s: s.str.split().str.len()).groupby(level=0).describe())

# ---- Q4 / Q5: duplicates --------------------------------------------------
print("\n=== DUPLICATE AUDIT (Q4, Q5) ===")
dup_exact = df["text_clean"].duplicated().sum()
df["p200"] = df["text_clean"].str[:200]
dup_near = df["p200"].duplicated().sum()
url_dup = raw["url"].duplicated().sum()
g = df.groupby("text_clean")["label_encoded"].nunique()
g2 = df.groupby("text_clean")["source"].nunique()
dup_stats = {
    "exact_duplicate_rows_after_preprocessing": int(dup_exact),
    "exact_duplicate_pct": round(100*dup_exact/len(df), 2),
    "near_duplicate_rows_200char_prefix": int(dup_near),
    "near_duplicate_pct": round(100*dup_near/len(df), 2),
    "duplicate_urls_in_raw_corpus": int(url_dup),
    "texts_with_conflicting_labels": int((g > 1).sum()),
    "texts_appearing_in_multiple_sources": int((g2 > 1).sum()),
}
for k, v in dup_stats.items(): print(f"  {k:44} {v}")

json.dump({"corpus": stats, "duplicates": dup_stats,
           "source_label_pure": bool(pure)},
          open(os.path.join(S, "out_dataset_stats.json"), "w"), indent=2)
print("\nwrote out_dataset_stats.json / out_source_composition.csv")
