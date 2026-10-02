# %% [markdown]
# # Batch 1·2·3 — 질문 1~5 EDA와 batch 비교
# Google Drive: MyDrive/skala_mini/archive 에 파일을 두고 위에서 아래로 실행하세요.
# GPU 불필요. 결과는 Colab 메모리에만 보관하며 CSV를 저장하지 않습니다.
# 과제 표의 mapping: Batch 1=2017-05-12, Batch 2=2018-02-20, Batch 3=2018-04-12.
# varcharge는 이번 3개 batch 분석에서 제외합니다.
# 실제 논문 메인 Batch 2 파일명(2017-06-30)과 과제의 mapping은 다릅니다.
# 이 노트북은 과제 mapping을 따르며 논문 재현 결과로 주장하지 않습니다.
#
# 기존 df_file 없이도 실행됩니다. summary 전체와 10·100사이클 Qdlin만 선택적으로 읽습니다.
# 각 batch는 별도로 유지하며, 질문 5의 모든 특성은 초기 100사이클에서만 계산합니다.
# label을 이용한 그룹 분할·상관계수는 EDA이며 예측 성능 평가가 아닙니다.

# %%
from pathlib import Path
from dotenv import load_dotenv
import os, re, importlib.util
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import display

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

load_dotenv(Path(__file__).resolve().parents[1]/'.env', override=False)
DATA_DIR = Path(os.environ.get(
    "BATTERY_DATA_DIR", "/content/drive/MyDrive/skala_mini/archive"
))
BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_COLORS = {"Batch 1": "#2563eb", "Batch 2": "#ea580c", "Batch 3": "#059669"}
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 110, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.alpha": 0.18, "legend.frameon": False
})
print("Data directory:", DATA_DIR)
for name, filename in BATCH_FILES.items():
    assert (DATA_DIR / filename).is_file(), f"Missing file: {filename}"
    print(name, filename)

# %% [markdown]
# ## 로딩과 품질 점검
# 첫 번째 0 placeholder는 실제로 모든 summary 신호가 0인 행만 제거합니다.
# Batch 1의 확인된 QDischarge 이상치: 배터리 1/12사이클, 배터리 19/40사이클.
# 이 두 값만 EDA 복사본에서 NaN으로 표시하고 원시값은 raw_summary에 보존합니다.
# 0·음수 용량, 저항, 충전시간은 EDA 복사본에서 NaN으로 표시하고 개수를 기록합니다.
# 온도의 0°C는 자동 제거하지 않습니다.
#
# 메인 Batch 1·3의 finite cycle_life는 기록 길이+1과 일치합니다.
# 기록 끝이 0.88Ah 임계값에 실제 도달한 것인지 확인되지 않으므로 제공 label을 사용하고
# 이를 관측된 EOL로 단정하지 않습니다. 데이터 audit에 이 관계를 보존합니다.

# %%
def ref(f, field, i):
    return f[field[i, 0]]

def text_value(obj):
    return "".join(chr(int(v)) for v in np.asarray(obj[()]).ravel(order="F"))

def at_cycle(df, col, n):
    values = df.loc[df["cycle"].eq(n), col]
    return float(values.iloc[0]) if len(values) else np.nan

def trend(df, col):
    x = df[["cycle", col]].dropna()
    if len(x) < 3 or x["cycle"].nunique() < 3:
        return np.nan
    return float(np.polyfit(x["cycle"], x[col], 1)[0])

summary_cols = ["cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]
batches, audit_rows = {}, []

for batch_name, filename in BATCH_FILES.items():
    cells = []
    with h5py.File(DATA_DIR / filename, "r") as f:
        b = f["batch"]
        for i in range(b["cycle_life"].shape[0]):
            s = ref(f, b["summary"], i)
            raw = pd.DataFrame({
                k: np.asarray(s[k][()]).reshape(-1).astype(float)
                for k in summary_cols
            })
            cleaned = raw.replace([np.inf, -np.inf], np.nan).copy()
            placeholder = cleaned.drop(columns="cycle").eq(0).all(axis=1)
            cleaned = cleaned.loc[~placeholder].copy()
            invalid_count = 0
            for k in ["QDischarge", "QCharge", "IR", "chargetime"]:
                invalid = cleaned[k].notna() & cleaned[k].le(0)
                invalid_count += int(invalid.sum())
                cleaned.loc[invalid, k] = np.nan

            removed_spikes = 0
            if batch_name == "Batch 1" and i + 1 in [1, 19]:
                spike_cycle = 12 if i + 1 == 1 else 40
                spike = cleaned["cycle"].eq(spike_cycle)
                removed_spikes = int(spike.sum())
                cleaned.loc[spike, "QDischarge"] = np.nan

            life_array = np.asarray(ref(f, b["cycle_life"], i)[()]).reshape(-1)
            life = float(life_array[0]) if life_array.size else np.nan
            if not np.isfinite(life) or life <= 0:
                life = np.nan
            policy = text_value(ref(f, b["policy_readable"], i))
            m = re.match(r"^([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", policy)
            c1, switch_soc, c2 = map(float, m.groups()) if m else [np.nan] * 3

            c = ref(f, b["cycles"], i)
            curve = {}
            for n in [10, 100]:
                positions = np.flatnonzero(raw["cycle"].to_numpy() == n)
                if len(positions) == 1:
                    curve[n] = np.asarray(
                        f[c["Qdlin"][int(positions[0]), 0]][()], dtype=float
                    ).reshape(-1)
            voltage = np.asarray(ref(f, b["Vdlin"], i)[()], dtype=float).reshape(-1)
            curve_ok = (
                all(n in curve for n in [10, 100])
                and len(voltage) == 1000
                and all(len(curve[n]) == 1000 for n in [10, 100])
                and np.isfinite(voltage).all()
                and all(np.isfinite(curve[n]).all() and not np.all(curve[n] == 0)
                        for n in [10, 100])
                and np.allclose(voltage, np.linspace(3.5, 2.0, 1000), atol=1e-6)
            )
            delta = curve[100] - curve[10] if curve_ok else None
            early = cleaned.loc[cleaned["cycle"].between(2, 100)].copy()
            ct = early["chargetime"].dropna()
            feat = {
                "C1": c1, "Switch_SOC": switch_soc, "C2": c2,
                "Qd_10": at_cycle(early, "QDischarge", 10),
                "Qd_100": at_cycle(early, "QDischarge", 100),
                "Qd_change_10_100": at_cycle(early, "QDischarge", 100) - at_cycle(early, "QDischarge", 10),
                "Qd_slope_early": trend(early, "QDischarge"),
                "IR_10": at_cycle(early, "IR", 10),
                "IR_change_10_100": at_cycle(early, "IR", 100) - at_cycle(early, "IR", 10),
                "Tavg_median_early": early["Tavg"].median(),
                "Tmax_max_early": early["Tmax"].max(),
                "Charge_time_median_early": ct.median(),
                "Charge_time_IQR_early": ct.quantile(.75) - ct.quantile(.25),
                "DQ_mean": np.mean(delta) if curve_ok else np.nan,
                "DQ_min": np.min(delta) if curve_ok else np.nan,
                "DQ_logvar": np.log10(max(np.var(delta), 1e-16)) if curve_ok else np.nan,
            }
            cells.append({
                "battery": i + 1, "batch": batch_name, "life": life, "policy": policy,
                "raw_summary": raw, "summary": cleaned,
                "voltage": voltage, "delta": delta, "features": feat,
            })
            q = raw["QDischarge"].to_numpy()
            audit_rows.append({
                "batch": batch_name, "battery": i + 1, "cycle_life": life,
                "recorded_cycles": len(raw),
                "label_equals_record_count_plus_one":
                    bool(np.isfinite(life) and np.isclose(life, len(raw) + 1)),
                "observed_below_088": bool(np.any(np.isfinite(q) & (q > 0) & (q < .88))),
                "placeholder_rows": int(placeholder.sum()),
                "invalid_positive_values": invalid_count,
                "capacity_spikes_masked": removed_spikes,
                "early_points": int(early["QDischarge"].notna().sum()),
                "charge_time_spikes_early": int((ct > 5 * ct.median()).sum()),
                "curve_ok": curve_ok, "policy_parse_ok": m is not None,
            })
    batches[batch_name] = cells
    print(f"Loaded {batch_name}: {len(cells)} cells")

audit = pd.DataFrame(audit_rows)
features = pd.DataFrame([
    {"batch": c["batch"], "battery": c["battery"], "policy": c["policy"],
     "cycle_life": c["life"], **c["features"]}
    for cells in batches.values() for c in cells
])
display(audit.groupby("batch").agg(
    cells=("battery", "size"), labels=("cycle_life", "count"),
    curve_ok=("curve_ok", "sum"), placeholder_rows=("placeholder_rows", "sum"),
    masked_capacity_spikes=("capacity_spikes_masked", "sum"),
    early_charge_spikes=("charge_time_spikes_early", "sum"),
    count_plus_one_labels=("label_equals_record_count_plus_one", "sum"),
    observed_below_088=("observed_below_088", "sum")
))
display(audit.loc[~audit["curve_ok"] | audit["cycle_life"].isna()
                  | audit["charge_time_spikes_early"].gt(0)])

# %% [markdown]
# ## 공통 그래프 함수
# 색은 제공 cycle_life를 뜻하며 missing label은 회색입니다.
# 질문 2의 후반 기울기와 knee는 전체 기록을 사용하는 진단값이고 모델 feature에 넣지 않습니다.
# knee는 연속 두 직선 모델의 SSE 개선율, 후반 기울기 가속 여부로 탐색한 후보입니다.
# 생물학적/전기화학적 수명 전환점을 확정한 값이 아닙니다.
# 15포인트 중앙값 평활화는 knee 탐색에만 사용하고 원시 용량 그래프는 그대로 그립니다.

# %%
all_life = features["cycle_life"].dropna()
life_norm = Normalize(float(all_life.min()), float(all_life.max()))
life_cmap = plt.get_cmap("viridis")

def life_color(c):
    return life_cmap(life_norm(c["life"])) if np.isfinite(c["life"]) else "#94a3b8"

def life_colorbar(fig, axes):
    fig.colorbar(ScalarMappable(norm=life_norm, cmap=life_cmap),
                 ax=axes, label="Provided cycle life", shrink=.8)

def heatmap(ax, table, title, cmap="coolwarm", vmax=1, annotate=True):
    vals = table.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(table.index)), table.index, fontsize=8)
    if annotate:
        for y in range(vals.shape[0]):
            for x in range(vals.shape[1]):
                if np.isfinite(vals[y, x]):
                    ax.text(x, y, f"{vals[y,x]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if abs(vals[y,x]) > .65 else "#111827")
    ax.set_title(title)
    ax.grid(False)
    return im

def knee_candidate(df):
    z = df[["cycle", "QDischarge"]].dropna().sort_values("cycle")
    z = z.loc[z["cycle"] >= 10].copy()
    if len(z) < 250:
        return {"knee": np.nan, "improvement": np.nan}
    y_full = z["QDischarge"].rolling(15, center=True, min_periods=5).median()
    x = z["cycle"].to_numpy()[::5]
    y = y_full.to_numpy()[::5]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    design = np.column_stack([np.ones(len(x)), x])
    linear = np.linalg.lstsq(design, y, rcond=None)[0]
    base_sse = np.sum((y - design @ linear) ** 2)
    if base_sse <= 1e-12:
        return {"knee": np.nan, "improvement": 0.0}
    best = None
    lower = max(x.min() + 100, np.quantile(x, .20))
    upper = min(x.max() - 100, np.quantile(x, .80))
    if lower >= upper:
        return {"knee": np.nan, "improvement": np.nan}
    for k in np.linspace(lower, upper, 40):
        design2 = np.column_stack([np.ones(len(x)), x, np.maximum(x - k, 0)])
        coef = np.linalg.lstsq(design2, y, rcond=None)[0]
        sse = np.sum((y - design2 @ coef) ** 2)
        if best is None or sse < best[0]:
            best = (sse, k, coef)
    sse, k, coef = best
    gain = 1 - sse / base_sse
    before, after = coef[1], coef[1] + coef[2]
    accepted = gain >= .30 and after < 0 and abs(after) >= 2 * max(abs(before), 1e-6)
    return {
        "knee": float(k) if accepted else np.nan,
        "best_breakpoint": float(k), "improvement": float(gain),
        "slope_before": float(before), "slope_after": float(after)
    }

knee_rows = []
for name, cells in batches.items():
    for c in cells:
        c["knee_info"] = knee_candidate(c["summary"])
        knee_rows.append({"batch": name, "battery": c["battery"],
                          "cycle_life": c["life"], **c["knee_info"]})
knee_table = pd.DataFrame(knee_rows)

# %% [markdown]
# ## 1. Cycle life 분포
# 150~2,300은 연구 전체 범위이고 현재 각 파일의 실제 범위와 다릅니다.
# short<500, long>1000 기준을 그대로 표시합니다. 500·1000은 middle에 포함합니다.
# missing label은 수명 분포에서 제외하고 개수를 별도로 표시합니다.
# 짧은 수명의 원인은 히스토그램만으로 설명할 수 없습니다.

# %%
def plot_q1(name):
    f = features.loc[features["batch"].eq(name)]
    y = f["cycle_life"].dropna()
    color = BATCH_COLORS[name]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].hist(y, bins=np.arange(100, 2401, 100), color=color, alpha=.8, edgecolor="white")
    axes[0].axvline(500, color="#dc2626", ls="--", label="Short: <500")
    axes[0].axvline(1000, color="#059669", ls="--", label="Long: >1000")
    axes[0].axvline(y.median(), color="#111827", lw=2, label=f"Median: {y.median():.0f}")
    axes[0].set(xlabel="Provided cycle life", ylabel="Cells", xlim=(100, 2400),
                title=f"{name} — Q1: life distribution")
    axes[0].legend(fontsize=8)
    counts = pd.Series({
        "Short <500": int((y < 500).sum()),
        "Middle 500–1000": int(y.between(500, 1000).sum()),
        "Long >1000": int((y > 1000).sum())
    })
    bars = axes[1].bar(counts.index, counts, color=["#dc2626", "#94a3b8", "#059669"])
    for bar, n in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, n + .4,
                     f"{n} / {len(y)}\n{n/len(y):.1%}", ha="center", fontsize=9)
    axes[1].set(ylim=(0, max(counts.max() * 1.25, 3)), ylabel="Cells",
                title=f"Finite labels: {len(y)} / {len(f)} | missing: {f['cycle_life'].isna().sum()}")
    axes[1].tick_params(axis="x", labelsize=9)
    plt.show()
    display(pd.DataFrame({
        "batch": [name], "finite_labels": [len(y)],
        "min": [y.min()], "median": [y.median()], "max": [y.max()],
        "short_pct": [(y < 500).mean()*100], "long_pct": [(y > 1000).mean()*100]
    }).round(2))

for name in batches:
    plot_q1(name)

# %% [markdown]
# ## 2. 열화 곡선과 가속·knee 후보
# 위 왼쪽: 실제 사이클별 방전용량. 색=수명이며 회색=수명 결측.
# 위 오른쪽: 10사이클 용량을 100%로 놓아 셀별 초기 용량 차이를 줄인 그래프.
# 아래 왼쪽: 초기 2~100 vs 마지막 100사이클 용량 기울기. y<x이면 후반이 더 음의 방향.
# 마지막 100사이클은 관측 기록의 끝이며 EOL 도달을 보장하지 않습니다.
# 아래 오른쪽: 후보 knee cycle과 제공 수명. 후보 없는 셀은 제외 수를 표시.
# 변화가 실험 조건 때문인지 실제 열화인지 이 모델만으로는 확정할 수 없습니다.
# raw Qd 확대 축 밖의 측정 개수도 함께 표시합니다.

# %%
def plot_q2(name):
    cells = batches[name]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    capacity_values, normalized_values, slope_rows = [], [], []
    for c in cells:
        d = c["summary"].sort_values("cycle")
        color = life_color(c)
        axes[0,0].plot(d["cycle"], d["QDischarge"], color=color, alpha=.65, lw=.85)
        q10 = at_cycle(d, "QDischarge", 10)
        if np.isfinite(q10) and q10 > 0:
            ratio = 100*d["QDischarge"]/q10
            axes[0,1].plot(d["cycle"], ratio, color=color, alpha=.65, lw=.85)
            normalized_values.extend(ratio.dropna().tolist())
        capacity_values.extend(d["QDischarge"].dropna().tolist())
        e = d.loc[d["cycle"].between(2, 100)]
        end = d.loc[d["cycle"] >= d["cycle"].max() - 99]
        se, sl = trend(e, "QDischarge"), trend(end, "QDischarge")
        if np.isfinite(se) and np.isfinite(sl):
            slope_rows.append((se, sl))
            axes[1,0].scatter(se*1000, sl*1000, color=color, s=28, alpha=.8)
        k = c["knee_info"].get("knee", np.nan)
        if np.isfinite(k) and np.isfinite(c["life"]):
            axes[1,1].scatter(k, c["life"], color=color, s=30)
    values = np.asarray(capacity_values)
    lo, hi = np.quantile(values, [.005, .995])
    margin = max((hi-lo)*.1, .005)
    outside = int(((values < lo-margin) | (values > hi+margin)).sum())
    axes[0,0].set(ylim=(lo-margin, hi+margin), xlabel="Cycle", ylabel="Capacity (Ah)",
                  title=f"{name} — Q2: discharge capacity")
    axes[0,0].text(.02,.04,f"Zoomed view: {outside} points outside axis",
                    transform=axes[0,0].transAxes, fontsize=8)
    if normalized_values:
        rlo, rhi = np.quantile(normalized_values, [.005,.995])
        axes[0,1].set_ylim(rlo-1, rhi+1)
    axes[0,1].axhline(100, color="#475569", ls="--", lw=1)
    axes[0,1].set(xlabel="Cycle", ylabel="Capacity / cycle-10 capacity (%)",
                  title="Relative capacity — observed record")
    if slope_rows:
        a = np.asarray(slope_rows)*1000
        lo2, hi2 = float(a.min()), float(a.max())
        axes[1,0].plot([lo2,hi2],[lo2,hi2], color="#475569", ls="--", label="Equal slope")
        axes[1,0].legend(fontsize=8)
    axes[1,0].set(xlabel="Early slope (mAh/cycle)", ylabel="Last-100 slope (mAh/cycle)",
                  title="Below diagonal: faster decline near record end")
    kt = knee_table.loc[knee_table["batch"].eq(name)]
    n_candidates = int(kt["knee"].notna().sum())
    axes[1,1].set(xlabel="Candidate knee cycle", ylabel="Provided cycle life",
                  title=f"Candidate knees: {n_candidates}/{len(cells)} — heuristic")
    life_colorbar(fig, list(axes.flat))
    plt.show()
    display(kt[["battery","cycle_life","knee","improvement","slope_before","slope_after"]].round(5))

for name in batches:
    plot_q2(name)

# %% [markdown]
# ## 3. ΔQ(V): Q100(V) − Q10(V)
# 실제 1,000포인트 전압축은 2.0~3.5V입니다.
# 절대 short<500 / long>1000 그룹 중 하나가 비면 해당 batch 내부의 수명 하위·상위 1/3로
# 비교를 전환하고 제목에 표시합니다. relative-low는 short<500과 다른 개념입니다.
# 개별 곡선, 그룹 평균±표준편차, log10 variance와 수명 scatter를 그립니다.
# log variance는 초기 100사이클에서 계산한 수명 예측 후보 feature입니다.

# %%
group_info = {}
def plot_q3(name):
    cells = [c for c in batches[name] if c["delta"] is not None and np.isfinite(c["life"])]
    lives = np.array([c["life"] for c in cells])
    short = [c for c in cells if c["life"] < 500]
    long = [c for c in cells if c["life"] > 1000]
    if not short or not long:
        q1, q2 = np.quantile(lives, [1/3, 2/3])
        low = [c for c in cells if c["life"] <= q1]
        high = [c for c in cells if c["life"] >= q2]
        low_label, high_label = f"Lower third (≤{q1:.0f})", f"Upper third (≥{q2:.0f})"
        definition = "Within-batch relative groups; absolute short/long group is empty"
    else:
        low, high = short, long
        low_label, high_label = "Short <500", "Long >1000"
        definition = "Absolute groups; middle cells omitted from group comparison"
    group_info[name] = {"definition":definition, "low_n":len(low), "high_n":len(high)}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    for c in cells:
        order = np.argsort(c["voltage"])
        axes[0].plot(c["voltage"][order], c["delta"][order], color=life_color(c), alpha=.55, lw=.9)
    axes[0].set(xlabel="Voltage (V)", ylabel="Q100 − Q10 (Ah)",
                title=f"{name} — Q3: ΔQ(V), n={len(cells)}")
    for group, label, color in [(low,low_label,"#dc2626"),(high,high_label,"#2563eb")]:
        a = np.vstack([c["delta"] for c in group])
        order = np.argsort(group[0]["voltage"])
        v = group[0]["voltage"][order]
        mean, sd = a.mean(axis=0)[order], a.std(axis=0)[order]
        axes[1].plot(v,mean,color=color,lw=2,label=f"{label}, n={len(group)}")
        axes[1].fill_between(v,mean-sd,mean+sd,color=color,alpha=.15)
    axes[1].set(xlabel="Voltage (V)", ylabel="ΔQ (Ah)", title="Group mean ± SD")
    axes[1].legend(fontsize=8)
    for c in cells:
        axes[2].scatter(c["features"]["DQ_logvar"],c["life"],color=life_color(c),s=30)
    f = features.loc[features["batch"].eq(name)]
    xy = f[["DQ_logvar","cycle_life"]].dropna()
    rho = xy.corr(method="spearman").iloc[0,1]
    axes[2].set(xlabel="log10 variance of ΔQ",ylabel="Provided cycle life",
                title=f"Early feature vs life | Spearman ρ={rho:.2f}, n={len(xy)}")
    life_colorbar(fig, [axes[0],axes[2]])
    plt.show()
    print(name, definition)
    display(pd.DataFrame([
        {"group":low_label,"n":len(low),"median_life":np.median([c["life"] for c in low])},
        {"group":high_label,"n":len(high),"median_life":np.median([c["life"] for c in high])}
    ]))

for name in batches:
    plot_q3(name)

# %% [markdown]
# ## 4. 충전 정책과 수명
# C1은 첫 충전 단계 C-rate, Switch_SOC는 전환 SOC, C2는 두 번째 C-rate입니다.
# 첫 패널: policy별 수명 평균과 ±1SD, 원시 셀 점, n.
# 나머지: C1/C2/전환 SOC와 수명의 관계.
# 반복 셀이 1~3개뿐인 policy가 많으므로 평균 순위는 불확실합니다.
# 여러 조건이 함께 바뀌어 "고속 충전 때문에 수명이 짧다"는 인과 결론은 낼 수 없습니다.
# Batch 2의 SLOWCYCLE 같은 실험 변형은 policy 문자열을 보존하며 임의로 동일 policy로 합치지 않습니다.

# %%
policy_tables = {}
def plot_q4(name):
    f = features.loc[features["batch"].eq(name)].copy()
    valid = f.dropna(subset=["cycle_life"])
    p = valid.groupby("policy")["cycle_life"].agg(["mean","std","count"]).sort_values("mean")
    policy_tables[name] = p
    fig = plt.figure(figsize=(16, max(8, len(p)*.32+2)), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3,1])
    ax = fig.add_subplot(gs[:,0])
    positions = np.arange(len(p))
    ax.errorbar(p["mean"],positions,xerr=p["std"].fillna(0),fmt="o",
                color=BATCH_COLORS[name],capsize=3,label="Mean ± SD")
    for j, policy in enumerate(p.index):
        ys = valid.loc[valid["policy"].eq(policy),"cycle_life"].to_numpy()
        jitter = np.linspace(-.10,.10,len(ys)) if len(ys)>1 else np.zeros(1)
        ax.scatter(ys,j+jitter,color="#334155",s=16,alpha=.65)
    ax.set_yticks(positions,[f"{policy}  [n={int(p.loc[policy,'count'])}]" for policy in p.index],fontsize=8)
    ax.set(xlabel="Provided cycle life",title=f"{name} — Q4: charging-policy life")
    ax.legend(fontsize=8)
    for row, col in enumerate(["C1","C2","Switch_SOC"]):
        sub = fig.add_subplot(gs[row,1])
        z = valid[[col,"cycle_life"]].dropna()
        sub.scatter(z[col],z["cycle_life"],color=BATCH_COLORS[name],alpha=.65,s=28)
        rho = z.corr(method="spearman").iloc[0,1] if z[col].nunique()>1 else np.nan
        sub.set(xlabel=col,ylabel="Provided cycle life",title=f"{col} vs life | ρ={rho:.2f}, n={len(z)}")
    plt.show()
    display(p.round(2))

for name in batches:
    plot_q4(name)

# %% [markdown]
# ## 5. 초기 특성–수명 상관과 다중공선성
# 각 배터리당 한 행의 features를 사용합니다. label 결측 셀은 수명 상관 계산에서 제외합니다.
# Spearman은 순위 상관입니다. 데이터 정규성 가정을 요구하지 않지만 관계의 원인을 증명하지 않습니다.
# 결측값은 EDA 상관계수 계산에서 쌍별 제외하며, 사용된 표본 수를 표시합니다.
# 최소 10개 유효 pair가 있어야 계산합니다.
# 표준화/결측 대체/특성 선택을 전체 batch에 fit하지 않습니다.
# corr ranking은 EDA 관찰이며 이를 보고 test 성능이 좋아지는 특성을 고르는 데 쓰지 않습니다.

# %%
feature_cols = [
    "C1","Switch_SOC","C2","Qd_10","Qd_100","Qd_change_10_100","Qd_slope_early",
    "IR_10","IR_change_10_100","Tavg_median_early","Tmax_max_early",
    "Charge_time_median_early","Charge_time_IQR_early",
    "DQ_mean","DQ_min","DQ_logvar"
]
corr_rankings, corr_matrices = {}, {}
def plot_q5(name):
    f = features.loc[features["batch"].eq(name)].copy()
    rankings = []
    for col in feature_cols:
        z = f[[col,"cycle_life"]].dropna()
        rho = z.corr(method="spearman",min_periods=10).iloc[0,1]
        rankings.append({"feature":col,"rho":rho,"n":len(z)})
    rank = pd.DataFrame(rankings).set_index("feature")
    corr_rankings[name] = rank
    shown = rank.dropna(subset=["rho"]).sort_values("rho")
    matrix = f[feature_cols].corr(method="spearman",min_periods=10)
    corr_matrices[name] = matrix
    fig, axes = plt.subplots(1,2,figsize=(18,7),layout="constrained",
                              gridspec_kw={"width_ratios":[1,1.8]})
    color = np.where(shown["rho"]>=0,"#2563eb","#dc2626")
    axes[0].barh(shown.index,shown["rho"],color=color,alpha=.8)
    axes[0].axvline(0,color="#475569",lw=1)
    axes[0].set(xlim=(-1,1),xlabel="Spearman correlation with life",
                title=f"{name} — Q5: early-feature association")
    axes[0].tick_params(axis="y",labelsize=8)
    im = heatmap(axes[1],matrix,"Feature–feature correlation",annotate=False)
    fig.colorbar(im,ax=axes[1],shrink=.8,label="Spearman ρ")
    plt.show()
    display(rank.assign(abs_rho=rank["rho"].abs()).sort_values("abs_rho",ascending=False).round(3))
    pairs = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i+1:]:
            r = matrix.loc[a,b]
            if np.isfinite(r) and abs(r)>=.85:
                pairs.append({"feature_1":a,"feature_2":b,"rho":r,
                              "n":len(f[[a,b]].dropna())})
    print("Strong feature pairs (|ρ| ≥ 0.85):")
    display(pd.DataFrame(pairs,columns=["feature_1","feature_2","rho","n"]).round(3))

for name in batches:
    plot_q5(name)

# %% [markdown]
# ## Batch 간 비교
# 각 셀에 같은 가중치를 줍니다. 수명이 긴 셀의 summary 행이 더 많은 문제를 피합니다.
# 비교 1: 수명, 충전 조건, 초기 ΔQ variance, 초기 용량 기울기, IR, 초기 온도.
# 비교 2: batch별 수명 상관계수의 일관성과 batch별 충전 정책 구성.
# Batch 2의 보조 실험 조건과 수명 결측, Batch 3의 newstructure suffix는 보존합니다.
# 분포가 다른 batch를 무작위로 섞으면 평가 해석이 어려워질 수 있습니다.
# 3개 batch를 모두 EDA하는 것은 과제 요구이며 모델 튜닝에 test label을 사용하는 것과 구분합니다.

# %%
compare_cols = [
    ("cycle_life","Provided cycle life"),
    ("C1","First-stage C-rate"),
    ("DQ_logvar","log10 variance of ΔQ"),
    ("Qd_slope_early","Early capacity slope (Ah/cycle)"),
    ("IR_10","Cycle-10 IR"),
    ("Tavg_median_early","Early median temperature (°C)")
]
names = list(batches)
rng = np.random.default_rng(42)
fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
for ax,(col,title) in zip(axes.flat,compare_cols):
    groups = [features.loc[features["batch"].eq(n),col].dropna().to_numpy() for n in names]
    bp = ax.boxplot(groups,tick_labels=names,patch_artist=True,showfliers=False)
    for patch,n in zip(bp["boxes"],names):
        patch.set_facecolor(BATCH_COLORS[n]);patch.set_alpha(.2)
    for j,(n,g) in enumerate(zip(names,groups),1):
        ax.scatter(j+rng.uniform(-.12,.12,len(g)),g,color=BATCH_COLORS[n],s=15,alpha=.65)
        ax.text(j,.98,f"n={len(g)}",transform=ax.get_xaxis_transform(),ha="center",va="top",fontsize=8)
    ax.set_title(title)
fig.suptitle("Batch comparison — one point per battery",fontsize=16,fontweight="bold")
plt.show()

rho_compare = pd.DataFrame({n:corr_rankings[n]["rho"] for n in names}).reindex(feature_cols)
fig,ax = plt.subplots(figsize=(7,8),layout="constrained")
im = heatmap(ax,rho_compare,"Early-feature vs life correlation by batch")
fig.colorbar(im,ax=ax,label="Spearman ρ",shrink=.8)
plt.show()
display(rho_compare.round(3))

policy_mix = features.groupby(["policy","batch"]).size().unstack(fill_value=0).reindex(columns=names,fill_value=0)
fig,ax = plt.subplots(figsize=(8,max(7,len(policy_mix)*.28)),layout="constrained")
im = ax.imshow(policy_mix.to_numpy(),cmap="Blues",aspect="auto",vmin=0)
ax.set_xticks(range(3),names)
ax.set_yticks(range(len(policy_mix)),policy_mix.index,fontsize=8)
for i in range(len(policy_mix)):
    for j in range(3):
        ax.text(j,i,str(policy_mix.iloc[i,j]),ha="center",va="center",fontsize=8)
ax.set_title("Charging-policy composition — all cells, including missing labels")
ax.grid(False)
fig.colorbar(im,ax=ax,label="Cells")
plt.show()

# %% [markdown]
# ## EDA 근거표와 모델 전략
# 아래 표를 보고 목표/조건을 결정합니다. 자동으로 test 성능에 맞춰 모델을 튜닝하지 않습니다.
# - Batch 1에서 초기 100사이클 특성으로 회귀 모델 후보를 개발합니다.
# - label 분포, 충전 정책, feature 분포가 다른 batch는 각각 별도 평가합니다.
# - short<500이 없는 Batch 1에서는 해당 기준의 이진 분류를 학습할 수 없습니다.
# - ΔQ 통계/초기 용량/저항/온도/충전조건 중 관련성이 있고 중복이 적은 후보를 고려합니다.
# - 강하게 연관된 특성은 선형 모델에서 중복을 줄이거나 Ridge/ElasticNet을 고려합니다.
# - 충전 정책별 표본이 적어 정책 평균 순위에 과도한 의미를 부여하지 않습니다.
# - test label을 이미 관찰한 평가 결과는 완전히 눈가림된 최종 평가라고 부르지 않습니다.
# - 제공 cycle_life의 측정종료 관계와 batch 실험 정의를 확인하기 전 수명 성능 주장은 탐색 수준입니다.

# %%
evidence_rows=[]
for n in names:
    f=features.loc[features["batch"].eq(n)]
    y=f["cycle_life"].dropna()
    r=corr_rankings[n].dropna(subset=["rho"])
    top=r["rho"].abs().idxmax() if len(r) else None
    evidence_rows.append({
        "batch":n,"cells":len(f),"labels":len(y),"missing_labels":f["cycle_life"].isna().sum(),
        "life_min":y.min(),"life_median":y.median(),"life_max":y.max(),
        "short_lt500":int((y<500).sum()),"long_gt1000":int((y>1000).sum()),
        "policies":f["policy"].nunique(),"knee_candidates":int(
            knee_table.loc[knee_table["batch"].eq(n),"knee"].notna().sum()),
        "largest_abs_rho_feature":top,"rho":r.loc[top,"rho"] if top else np.nan
    })
evidence=pd.DataFrame(evidence_rows)
display(evidence.round(3))
print("Colab memory objects: batches, audit, features, knee_table, policy_tables, corr_rankings, evidence")
print("EDA completed. No CSV files saved.")
