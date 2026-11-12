import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

ROOT = Path(__file__).resolve().parent.parent
DATA, PROCESSED, RESULTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "results"
OUT = ROOT / "docs" / "data.js"

SEED, DEV_FRAC, SPLIT = 171, 0.2, 0
DEMO_N = [16, 64, 256, 1024]
MODELS = {
    "nb": lambda: BernoulliNB(alpha=0.01, binarize=None),                      # bnb_a001 in the notebook
    "lr": lambda: LogisticRegression(C=1e4, tol=1e-8, max_iter=10_000),        # lr in the notebook
}
SUMMARY_NAME = {"nb": "bnb_a001", "lr": "lr"}
TEST_SIZE = {"sms": 1000, "enron": 2000}
LABEL = {"sms": "SMS Spam Collection", "enron": "Enron-Spam emails"}
SAMPLES_PER_CLASS, SAMPLE_MAX_CHARS = 15, 700
VOCAB_PREVIEW = 25          # top-ranked words whose mutual information is shown on the page

TOKEN_RE = re.compile(r"[a-z0-9]+|[£$€!]")
# spam corpora contain adult content; keep it off the presentation screen
NOT_FOR_CLASS = re.compile(r"sex|porn|xxx|adult|nude|naked|horny|cum|ass|fuck|shit|bitch|dick|cock|pussy|slut|teen|"
                           r"viagra|cialis|penis|erect|breast|boob|orgasm|erotic|hot\s|babe|girl|wom[ae]n|lolita|incest|"
                           r"rape|kill|drug|pill|meds|pharm|dating|lonely|escort|18\+|wife|wives|virgin|"
                           r"dogg|laid|gigolo|rection|xual|ialis|iagr|gra\b|xan|n @ x|vic ' od|presc|medic|tablet|"
                           r"propec|bald|ladies|se \* x|climax|enlarge|inch|weight|diet", re.I)


# the notebook's pipeline, copied so the pool rows can be matched back to their text
def read_sms():
    path = next(p for p in (DATA / "SMSSpamCollection", DATA / "SMSSpamCollection.csv") if p.exists())
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            label, text = line.split("\t", 1)
            rows.append((text, label == "spam"))
    return rows


def read_enron():
    df = pd.read_csv(DATA / "enron_spam_data.csv")
    text = df["Subject"].fillna("") + "\n" + df["Message"].fillna("")
    return list(zip(text, df["Spam/Ham"] == "spam"))


def load_corpus(reader):
    texts, labels, seen = [], [], set()
    for text, is_spam in reader():
        key = " ".join(text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        labels.append(int(is_spam))
    return texts, np.array(labels, dtype=int)


def stratified_dev_split(y, dev_frac, seed):
    rng = np.random.default_rng(seed)
    dev = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        dev.append(idx[: round(dev_frac * len(idx))])
    dev = np.sort(np.concatenate(dev))
    return dev, np.setdiff1d(np.arange(len(y)), dev)


def m_grid(m_min, m_max, per_octave):
    n_points = int(np.floor(per_octave * np.log2(m_max / m_min))) + 1
    grid = np.unique(np.round(m_min * 2.0 ** (np.arange(n_points) / per_octave)).astype(int))
    return grid if grid[-1] == m_max else np.append(grid, m_max)


class Split:
    def __init__(self, y, split, test_size, seed):
        rng = np.random.default_rng([seed, split])
        self.p_spam = y.mean()
        n_test_spam = round(test_size * self.p_spam)
        spam = rng.permutation(np.flatnonzero(y == 1))
        ham = rng.permutation(np.flatnonzero(y == 0))
        self.test = np.sort(np.concatenate([spam[:n_test_spam], ham[: test_size - n_test_spam]]))
        self._spam, self._ham = spam[n_test_spam:], ham[test_size - n_test_spam:]

    def train(self, m):
        k = int(np.clip(round(m * self.p_spam), 1, m - 1))
        k = max(min(k, len(self._spam)), m - len(self._ham))
        return np.concatenate([self._spam[:k], self._ham[: m - k]])


# export
def linear_rule(model, n):
    """(bias, weights) with log-odds = bias + sum of weights over the words present."""
    if isinstance(model, LogisticRegression):
        return float(model.intercept_[0]), model.coef_[0]
    present = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    absent = np.log1p(-np.exp(model.feature_log_prob_[1])) - np.log1p(-np.exp(model.feature_log_prob_[0]))
    bias = model.class_log_prior_[1] - model.class_log_prior_[0] + absent.sum()
    return float(bias), present - absent


def export(dataset, reader):
    ranked = json.loads((PROCESSED / dataset / "vocab.json").read_text(encoding="utf-8"))
    vocab = [v["token"] for v in ranked]
    X_counts = sp.load_npz(PROCESSED / dataset / "pool_counts.npz").tocsc()
    y = np.load(PROCESSED / dataset / "pool_labels.npy")
    X = (X_counts > 0).astype(np.float64).tocsc()

    texts, y_all = load_corpus(reader)
    _, pool = stratified_dev_split(y_all, DEV_FRAC, SEED)
    assert np.array_equal(y_all[pool], y), "pool labels do not match data/processed"
    col = {t: j for j, t in enumerate(vocab)}
    for i in np.random.default_rng(0).choice(len(pool), 300, replace=False):     # text <-> row alignment
        cols = sorted({col[t] for t in TOKEN_RE.findall(texts[pool[i]].lower()) if t in col})
        assert cols == sorted(X[i].nonzero()[1].tolist()), "pool rows do not match the raw text"

    split = Split(y, SPLIT, TEST_SIZE[dataset], SEED)
    grid = [int(m) for m in m_grid(8, len(y) - TEST_SIZE[dataset], 2)]
    summary = pd.read_csv(RESULTS / dataset / "summary_test_error.csv")
    assert sorted(summary.m.unique()) == grid, "training sizes differ from the saved results"

    out = {"label": LABEL[dataset], "m": grid, "vocab": vocab[: max(DEMO_N)],
           "mi": [round(v["mi"], 4) for v in ranked[:VOCAB_PREVIEW]], "n": {}}
    for n in DEMO_N:
        Xn = X[:, :n].tocsr()
        entry = {"err": {}, "ci": {}, "w": {k: [] for k in MODELS}, "split_err": {k: [] for k in MODELS}}
        for m in grid:
            tr = split.train(m)
            for key, make in MODELS.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = make().fit(Xn[tr], y[tr])
                bias, w = linear_rule(model, n)
                z = bias + Xn[split.test] @ w
                ref = (model.decision_function(Xn[split.test]) if key == "lr" else
                       np.diff(model.predict_joint_log_proba(Xn[split.test]), axis=1).ravel())
                assert np.allclose(z, ref, atol=1e-6), "linear rule does not reproduce the model"
                entry["w"][key].append([round(bias, 3)] + [round(float(v), 3) for v in w])
                entry["split_err"][key].append(round(float(np.mean((z > 0) != (y[split.test] == 1))), 4))
        for key, name in SUMMARY_NAME.items():
            s = summary[(summary.n == n) & (summary.model == name)].set_index("m").loc[grid]
            entry["err"][key] = [round(float(v), 5) for v in s["mean"]]
            entry["ci"][key] = [round(float(v), 5) for v in s["ci95"]]
        out["n"][str(n)] = entry
        print(f"  {dataset} n={n}: split-0 error at largest m  NB {entry['split_err']['nb'][-1]:.3f}  LR {entry['split_err']['lr'][-1]:.3f}"
              f"   (200-split mean  NB {entry['err']['nb'][-1]:.3f}  LR {entry['err']['lr'][-1]:.3f})")

    # held-out test messages of split 0: no model on the page was trained on them
    rng = np.random.default_rng(SEED)
    samples = []
    for label in (1, 0):
        idx = [i for i in split.test if y[i] == label and 60 <= len(texts[pool[i]].strip()) <= SAMPLE_MAX_CHARS
               and not NOT_FOR_CLASS.search(texts[pool[i]])]
        for i in rng.choice(idx, SAMPLES_PER_CLASS, replace=False):
            samples.append({"text": texts[pool[i]].strip(), "spam": int(label)})
    out["samples"] = samples
    return out


if __name__ == "__main__":
    demo = {"sms": export("sms", read_sms), "enron": export("enron", read_enron)}
    OUT.write_text("window.DEMO = " + json.dumps(demo, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.2f} MB)")
