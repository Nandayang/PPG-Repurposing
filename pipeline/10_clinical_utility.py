"""Calibration, prevalence correction, operating points, age standardization and decision curves
(functions imported by 21_paper_analyses.py; main() runs them on a 10-fold OOF file).
"""
import os, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

CACHE   = os.path.join(os.environ.get("PPG_ROOT", "."), "cache")
DERIVED = os.path.join(os.environ.get("PPG_ROOT", "."), "derived")
OUT     = os.path.join(os.environ.get("PPG_ROOT", "."), "results")
FIG     = os.path.join(os.environ.get("PPG_ROOT", "."), "figures")
os.makedirs(FIG, exist_ok=True)
EPS = 1e-6
AGE_BINS  = [0, 50, 60, 70, 200]
AGE_NAMES = ["<50", "50-60", "60-70", ">70"]

def logit(p): return np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
def expit(z): return 1 / (1 + np.exp(-z))

def rebuild_ids(task, mode, quality, folds, y_ref):
    idx  = pd.read_csv(os.path.join(CACHE, "index.csv"))
    suf  = "" if mode == "phase" else "_aligned"
    meta = pd.read_csv(os.path.join(CACHE, f"beat_coh{suf}_meta.csv"))
    pos, neg = {"cirrhosis": ("lbl_cirrhosis", "lbl_control"),
                "cld": ("lbl_cld", "lbl_control"),
                "phtn": ("lbl_phtn", "lbl_control"),
                "decomp": ("lbl_cirr_decomp", "lbl_cirr_comp"),
                **{f"spec_{k}": (f"lbl_spec_{k}_pos", f"lbl_spec_{k}_neg")
                   for k in ["hf","sepsis","esrd","any"]}}[task]
    m = (meta.ok_beat == 1).to_numpy() & ((idx[pos] == 1) | (idx[neg] == 1)).to_numpy()
    if not (task.startswith("spec_") or task == "decomp"):
        m = m & ~(((idx[pos] == 0) & (idx[neg] == 0)).to_numpy())
    if quality == "both":
        m = m & (idx.both_good == 1).to_numpy()
    d = idx.loc[m].copy()
    d["_y"] = (d[pos] == 1).astype(int)
    sids, ys = [], []
    for f in range(folds):
        dd = d[d.strat_fold == f]
        if len(dd) == 0 or dd._y.nunique() < 2: continue
        g = dd.groupby("subject_id").agg(y=("_y", "max"))
        sids += list(g.index); ys += list(g.y.to_numpy())
    ys = np.asarray(ys)
    assert len(sids) == len(y_ref), f"{len(sids)} patients rebuilt, {len(y_ref)} in the OOF file"
    assert np.array_equal(ys, y_ref.astype(int)), "labels do not match the OOF file"
    pt = d.groupby("subject_id").agg(age=("age", "median"),
                                     gender=("gender", "first")).reset_index()
    out = pd.DataFrame({"subject_id": sids}).merge(pt, on="subject_id", how="left")
    return out

def cross_calibrate(p, y, fold, kind="platt"):
    z = logit(p); out = np.zeros_like(p)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if len(np.unique(y[tr])) < 2: out[te] = p[te]; continue
        if kind == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(z[tr, None], y[tr])
            out[te] = m.predict_proba(z[te, None])[:, 1]
        else:
            m = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = np.clip(m.predict(p[te]), EPS, 1-EPS)
    return out

def cal_metrics(p, y, nbin=10):
    br = brier_score_loss(y, p)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    slope, inter = float(m.coef_[0][0]), float(m.intercept_[0])
    q = np.quantile(p, np.linspace(0, 1, nbin+1)); q[0] -= 1e-9
    b = np.digitize(p, q[1:-1]); ece = 0.0; pts = []
    for k in range(nbin):
        s = b == k
        if s.sum() < 5: continue
        ece += s.mean() * abs(p[s].mean() - y[s].mean())
        pts.append([float(p[s].mean()), float(y[s].mean()), int(s.sum())])
    return dict(brier=float(br), slope=slope, intercept=inter, ece=float(ece), bins=pts)

def op_table(y, p, prev, boot=2000, seed=0):
    fpr, tpr, th = roc_curve(y, p)
    picks = [("Sens 90%", int(np.argmax(tpr >= 0.90))),
             ("Sens 80%", int(np.argmax(tpr >= 0.80))),
             ("Youden",   int(np.argmax(tpr - fpr))),
             ("Spec 90%", int(np.argmax(fpr >= 0.10))),
             ("Spec 95%", int(np.argmax(fpr >= 0.05)))]
    rng = np.random.default_rng(seed); rows = []
    for nm, i in picks:
        t = th[i]
        se, sp = float(tpr[i]), float(1 - fpr[i])
        bs = []
        for _ in range(boot):
            j = rng.integers(0, len(y), len(y))
            yy, pp = y[j], p[j]
            if yy.sum() < 2 or (1-yy).sum() < 2: continue
            bs.append([(pp[yy == 1] >= t).mean(), (pp[yy == 0] < t).mean()])
        bs = np.array(bs) if bs else np.full((1, 2), np.nan)
        se_ci = np.percentile(bs[:, 0], [2.5, 97.5]); sp_ci = np.percentile(bs[:, 1], [2.5, 97.5])
        ppv = se*prev / max(se*prev + (1-sp)*(1-prev), EPS)
        npv = sp*(1-prev) / max(sp*(1-prev) + (1-se)*prev, EPS)
        pv = bs[:, 0]*prev / np.maximum(bs[:, 0]*prev + (1-bs[:, 1])*(1-prev), EPS)
        rows.append(dict(name=nm, thr=float(t), sens=se, sens_ci=list(se_ci),
                         spec=sp, spec_ci=list(sp_ci), ppv=float(ppv),
                         ppv_ci=list(np.percentile(pv, [2.5, 97.5])),
                         npv=float(npv), nns=float(1/max(ppv, EPS))))
    return rows

def age_standardised(y, p, age, thr, coh):
    ab = np.digitize(age, AGE_BINS[1:-1])
    cb = np.digitize(coh.age.to_numpy(), AGE_BINS[1:-1])
    wc = np.array([(cb[coh.lbl_cirrhosis == 1] == k).mean() for k in range(len(AGE_NAMES))])
    wn = np.array([(cb[coh.lbl_cirrhosis == 0] == k).mean() for k in range(len(AGE_NAMES))])
    se_k, sp_k, n_k = [], [], []
    for k in range(len(AGE_NAMES)):
        s = ab == k
        pos, neg = s & (y == 1), s & (y == 0)
        se_k.append((p[pos] >= thr).mean() if pos.sum() >= 5 else np.nan)
        sp_k.append((p[neg] <  thr).mean() if neg.sum() >= 5 else np.nan)
        n_k.append([int(pos.sum()), int(neg.sum())])
    se_k, sp_k = np.array(se_k), np.array(sp_k)
    ok = ~np.isnan(se_k) & ~np.isnan(sp_k)
    se_std = float(np.sum(wc[ok]*se_k[ok]) / np.sum(wc[ok]))
    sp_std = float(np.sum(wn[ok]*sp_k[ok]) / np.sum(wn[ok]))
    return dict(strata=AGE_NAMES, sens=[None if np.isnan(v) else float(v) for v in se_k],
                spec=[None if np.isnan(v) else float(v) for v in sp_k], n=n_k,
                w_case=[float(v) for v in wc], w_ctrl=[float(v) for v in wn],
                sens_std=se_std, spec_std=sp_std)

def dca(y, p, prev, pts):
    fpr, tpr, th = roc_curve(y, p)
    rows = []
    for pt in pts:
        i = int(np.argmin(np.abs(th - pt)))
        se, sp = float(tpr[i]), float(1 - fpr[i])
        nb  = se*prev - (1-sp)*(1-prev) * (pt/(1-pt))
        nba = prev - (1-prev) * (pt/(1-pt))
        rows.append(dict(pt=float(pt), nb_model=float(nb), nb_all=float(nba), nb_none=0.0,
                         sens=se, spec=sp, per_1000=float(nb*1000)))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=os.path.join("results", "oof_cirrhosis_beat2d_aligned_init-C_A_r0.7_e50.npz"))
    ap.add_argument("--tag", default="ssl")
    ap.add_argument("--task", default="cirrhosis")
    ap.add_argument("--mode", default="aligned", choices=["phase", "aligned"])
    ap.add_argument("--quality", default="none", choices=["none", "both"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--prev", type=float, default=-1, help="target prevalence (default: source population)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--calibrator", default="platt", choices=["platt","isotonic","auto"],
                    help="auto: lower Brier score")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args()

    z = np.load(os.path.join(os.environ.get("PPG_ROOT", "."), a.oof) if not os.path.isabs(a.oof) else a.oof)
    y, p = z["y"].astype(int), z["p"].astype(float)
    coh = pd.read_csv(os.path.join(DERIVED, "cohort_patients.csv"))
    prev_src = float((coh.lbl_cirrhosis == 1).mean())
    prev = prev_src if a.prev < 0 else a.prev
    pc = float(y.mean())

    info = rebuild_ids(a.task, a.mode, a.quality, a.folds, y)
    idx = pd.read_csv(os.path.join(CACHE, "index.csv"))
    fold = info.merge(idx.groupby("subject_id").strat_fold.first().reset_index(),
                      on="subject_id", how="left").strat_fold.to_numpy()

    print("="*74)
    print(f" Clinical utility   {a.tag}   {a.oof}")
    print("="*74)
    print(f"[cohort] {len(y):,} patients, {int(y.sum())} positive ({pc:.1%})")
    print(f"[source] {len(coh):,} ICU patients with PPG, "
          f"{int((coh.lbl_cirrhosis==1).sum())} with cirrhosis ({prev_src:.2%})")
    print(f"[target prevalence] {prev:.2%}   AUC {roc_auc_score(y,p):.4f}")

    print("\n" + "-"*74); print(" 1. Calibration (fitted on the other folds)"); print("-"*74)
    cals = {"uncalibrated": p, "Platt": cross_calibrate(p, y, fold, "platt"),
            "Isotonic": cross_calibrate(p, y, fold, "isotonic")}
    cm = {k: cal_metrics(v, y) for k, v in cals.items()}
    print(f"{'':<13}{'Brier':>9}{'slope':>10}{'intercept':>10}{'ECE':>9}{'AUC':>9}")
    for k, v in cm.items():
        print(f"{k:<13}{v['brier']:>9.4f}{v['slope']:>10.3f}{v['intercept']:>10.3f}"
              f"{v['ece']:>9.4f}{roc_auc_score(y, cals[k]):>9.4f}")
    if a.calibrator == "auto":
        best = min(("Platt", "Isotonic"), key=lambda k: cm[k]["brier"])
    else:
        best = {"platt": "Platt", "isotonic": "Isotonic"}[a.calibrator]
    pcal = cals[best]
    print(f"  calibrator: {best}")

    print("\n" + "-"*74); print(" 2. Prevalence correction (intercept shift)"); print("-"*74)
    shift = np.log(prev/(1-prev)) - np.log(pc/(1-pc))
    padj = expit(logit(pcal) + shift)
    print(f"  cohort {pc:.3f} -> target {prev:.3f}   logit shift {shift:+.4f}")
    print(f"  median probability {np.median(padj):.3f}  "
          f"cases {np.median(padj[y==1]):.3f}  controls {np.median(padj[y==0]):.3f}")

    print("\n" + "-"*74); print(f" 3. Operating points (prevalence {prev:.1%}, {a.boot} bootstrap resamples)"); print("-"*74)
    ops = op_table(y, padj, prev, boot=a.boot)
    print(f"{'point':<11}{'sensitivity':>19}{'specificity':>19}{'PPV':>17}{'NNS':>7}{'missed':>7}")
    for r in ops:
        print(f"{r['name']:<11}"
              f"{r['sens']:>7.3f} [{r['sens_ci'][0]:.3f},{r['sens_ci'][1]:.3f}]"
              f"{r['spec']:>7.3f} [{r['spec_ci'][0]:.3f},{r['spec_ci'][1]:.3f}]"
              f"{r['ppv']:>7.1%} [{r['ppv_ci'][0]:.1%},{r['ppv_ci'][1]:.1%}]"
              f"{r['nns']:>7.1f}{1-r['sens']:>7.1%}")

    print("\n" + "-"*74); print(" 3b. Age standardization"); print("-"*74)
    yo = [r for r in ops if r["name"] == "Youden"][0]
    st = age_standardised(y, padj, info.age.to_numpy(), yo["thr"], coh)
    print(f"  Youden threshold {yo['thr']:.3f}")
    print(f"{'age':<9}{'n+':>6}{'n-':>7}{'sens':>9}{'spec':>9}{'w cases':>12}{'w controls':>12}")
    for k, nm in enumerate(st["strata"]):
        se = st["sens"][k]; sp = st["spec"][k]
        print(f"{nm:<9}{st['n'][k][0]:>6}{st['n'][k][1]:>7}"
              f"{(f'{se:.3f}' if se is not None else '  -'):>9}"
              f"{(f'{sp:.3f}' if sp is not None else '  -'):>9}"
              f"{st['w_case'][k]:>12.1%}{st['w_ctrl'][k]:>12.1%}")
    ppv_std = st["sens_std"]*prev / max(st["sens_std"]*prev + (1-st["spec_std"])*(1-prev), EPS)
    print(f"  crude         sens {yo['sens']:.3f}  spec {yo['spec']:.3f}  PPV {yo['ppv']:.1%}  NNS {yo['nns']:.1f}")
    print(f"  standardized  sens {st['sens_std']:.3f}  spec {st['spec_std']:.3f}  "
          f"PPV {ppv_std:.1%}  NNS {1/max(ppv_std,EPS):.1f}")

    print("\n" + "-"*74); print(" 4. Decision curve (net benefit at the target prevalence)"); print("-"*74)
    dc = dca(y, padj, prev, [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30])
    print(f"{'threshold':>9}{'model':>10}{'all':>10}{'none':>7}{'per 1000':>12}")
    for r in dc:
        print(f"{r['pt']:>9.3f}{r['nb_model']:>10.4f}{r['nb_all']:>10.4f}{0:>7.1f}"
              f"{r['per_1000']:>12.1f}")
    good = [r["pt"] for r in dc if r["nb_model"] > max(r["nb_all"], 0)]
    if good:
        print(f"  model better than both strategies at thresholds {min(good):.3f}-{max(good):.3f}")

    print("\n" + "-"*74); print(" 5. NNS by prevalence"); print("-"*74)
    pv_list = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    print(f"{'prev':>8}" + "".join(f"{r['name']:>13}" for r in ops))
    for pr in pv_list:
        cells = []
        for r in ops:
            v = r["sens"]*pr / max(r["sens"]*pr + (1-r["spec"])*(1-pr), EPS)
            cells.append(f"{1/max(v,EPS):>13.1f}")
        print(f"{pr:>8.1%}" + "".join(cells))

    res = dict(tag=a.tag, oof=a.oof, n=len(y), n_pos=int(y.sum()), prev_cohort=pc,
               prev_target=prev, prev_source=prev_src, auc=float(roc_auc_score(y, p)),
               calibration={k: cm[k] for k in cm}, calibrator=best, logit_shift=float(shift),
               operating_points=ops, age_standardised=st, dca=dc)
    jp = os.path.join(OUT, f"clinical_utility_{a.tag}.json")
    json.dump(res, open(jp, "w"), indent=2)
    pd.DataFrame({"subject_id": info.subject_id, "age": info.age, "gender": info.gender,
                  "y": y, "p_raw": p, "p_cal": pcal, "p_adj": padj}).to_csv(
        os.path.join(OUT, f"patient_predictions_{a.tag}.csv"), index=False)
    print(f"\n[saved] {jp}")

    if not a.no_fig:
        try:
            make_fig(y, p, pcal, padj, cm, best, ops, dc, prev, pv_list, a.tag)
            print(f"[saved] figures/clinical_utility_{a.tag}.png")
        except Exception as e:
            print(f"[!] figure failed: {e}")

def make_fig(y, p, pcal, padj, cm, best, ops, dc, prev, pv_list, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0,0].plot(fpr, tpr, lw=2, color="#1f4e79")
    ax[0,0].plot([0,1],[0,1],"--",color="#999",lw=1)
    for r in ops:
        ax[0,0].plot(1-r["spec"], r["sens"], "o", ms=6, color="#c0392b")
        ax[0,0].annotate(r["name"], (1-r["spec"], r["sens"]), textcoords="offset points",
                         xytext=(6,-9), fontsize=8)
    ax[0,0].set_xlabel("1 - Specificity"); ax[0,0].set_ylabel("Sensitivity")
    ax[0,0].set_title(f"ROC (patient-level OOF), AUC = {roc_auc_score(y,p):.3f}")

    for nm, col in [("uncalibrated","#aaaaaa"), (best,"#1f4e79")]:
        b = np.array(cm[nm]["bins"])
        ax[0,1].plot(b[:,0], b[:,1], "o-", color=col, ms=5,
                     label=f"{nm} (Brier {cm[nm]['brier']:.4f})")
    ax[0,1].plot([0,1],[0,1],"--",color="#999",lw=1)
    ax[0,1].set_xlabel("Predicted probability"); ax[0,1].set_ylabel("Observed frequency")
    ax[0,1].set_title("Calibration (cross-fold)"); ax[0,1].legend(fontsize=8)

    pts = np.array([r["pt"] for r in dc]); nbm = np.array([r["nb_model"] for r in dc])
    nba = np.array([r["nb_all"] for r in dc])
    ax[1,0].plot(pts, nbm, "-o", color="#1f4e79", lw=2, ms=4, label="Model")
    ax[1,0].plot(pts, nba, "--", color="#c0392b", lw=1.5, label="Screen all")
    ax[1,0].axhline(0, color="#333", lw=1, label="Screen none")
    ax[1,0].set_ylim(min(-0.02, nbm.min()*1.1), max(nbm.max()*1.3, 0.02))
    ax[1,0].set_xlabel("Threshold probability"); ax[1,0].set_ylabel("Net benefit")
    ax[1,0].set_title(f"Decision curve (prevalence {prev:.1%})"); ax[1,0].legend(fontsize=8)

    for r in ops:
        nns = [1/max(r["sens"]*pr/(r["sens"]*pr+(1-r["spec"])*(1-pr)), 1e-9) for pr in pv_list]
        ax[1,1].plot(np.array(pv_list)*100, nns, "-o", ms=4, label=r["name"])
    ax[1,1].axvline(prev*100, color="#c0392b", ls="--", lw=1)
    ax[1,1].set_xlabel("Cirrhosis prevalence (%)"); ax[1,1].set_ylabel("Number needed to screen")
    ax[1,1].set_title("NNS vs prevalence"); ax[1,1].legend(fontsize=8)
    for a_ in ax.ravel(): a_.grid(alpha=.25)
    fig.suptitle(f"Clinical utility ({tag})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"clinical_utility_{tag}.png"), dpi=150)

if __name__ == "__main__":
    warnings.filterwarnings("ignore"); main()
