import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join("src"))
from eval_test_sets import load_test_set, build_test_features, TEST_ROOT
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE = "."
merged = pd.read_pickle(os.path.join("data","processed","merged_all.pkl"))
features = pd.read_pickle(os.path.join("data","processed","features_all.pkl"))
X = features.values.astype(np.float64)
train_mask = (merged["split"]=="train").values
val_mask = (merged["split"]=="val").values
train_seqs = set(merged.loc[train_mask,"seq21"].tolist())
rng = np.random.RandomState(0)
tr_idx = rng.choice(np.where(train_mask)[0], 6000, replace=False)
va_idx = rng.choice(np.where(val_mask)[0], 3000, replace=False)
scaler = StandardScaler().fit(X[tr_idx])
pca = PCA(n_components=20, random_state=0).fit(scaler.transform(X[tr_idx]))
def embed(idx):
    return pca.transform(scaler.transform(X[idx]))
Ztr = embed(tr_idx); Zva = embed(va_idx)

def rbf_mmd(A, B, sigma=4.0):
    # 无偏 MMD^2 估计（RBF）
    n = A.shape[0]; m = B.shape[0]
    A2 = (A**2).sum(1, keepdims=True); B2 = (B**2).sum(1, keepdims=True)
    Kaa = np.exp(-(A2 + A2.T - 2*A@A.T)/(2*sigma**2))
    Kbb = np.exp(-(B2 + B2.T - 2*B@B.T)/(2*sigma**2))
    Kab = np.exp(-(A2 + B2.T - 2*A@B.T)/(2*sigma**2))
    np.fill_diagonal(Kaa, 0); np.fill_diagonal(Kbb, 0)
    return Kaa.sum()/(n*(n-1)) + Kbb.sum()/(m*(m-1)) - 2*Kab.mean()

ext_res = json.load(open(os.path.join("results","tables","independent_test_results.json")))
rows = []
for d in sorted(os.listdir(TEST_ROOT)):
    dp = os.path.join(TEST_ROOT, d)
    if not os.path.isdir(dp): continue
    seqs, labels, shape = load_test_set(dp)
    keep = [i for i,s in enumerate(seqs) if s not in train_seqs]
    seqs_k = [seqs[i] for i in keep]; shape_k = {sh:v[keep] for sh,v in shape.items()}
    df_f = build_test_features(seqs_k, shape_k)
    Xt = df_f.values.astype(np.float64)
    Zt = pca.transform(scaler.transform(Xt))
    mmd = rbf_mmd(Ztr[:2000], Zt[:2000])
    sp = ext_res[d]["WT"]["spearman"]
    rows.append((d, mmd, sp))
    print(f"{d:20s} MMD={mmd:.5f}  spearman={sp:.3f}")
rho, p = spearmanr([r[1] for r in rows],[r[2] for r in rows])
print(f"Spearman(MMD, spearman) = {rho:.4f} (p={p:.4f})")
rho5, p5 = spearmanr([r[1] for r in rows if r[0]!="Ren_Ff_2015"],[r[2] for r in rows if r[0]!="Ren_Ff_2015"])
print(f"excl Ren: {rho5:.4f} (p={p5:.4f})")
# val anchor
mmd_val = rbf_mmd(Ztr[:2000], Zva[:2000])
print(f"VAL MMD anchor = {mmd_val:.5f}")
