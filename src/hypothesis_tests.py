"""기존 EDA의 features, batches에 적용. CSV 저장 없이 메모리에 결과 반환."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_hypothesis_tests(features, batches, seed=20261001,
                         permutations=19999, bootstrap_samples=5000):
    # 같은 충전 정책의 반복 셀은 정책별 중앙값 한 점으로 집계.
    train = features.loc[features["batch"].eq("Batch 1")].copy()
    policy_table = train.groupby("policy").agg(
        cycle_life=("cycle_life", "median"),
        DQ_logvar=("DQ_logvar", "median"),
        C1=("C1", "first"), C2=("C2", "first"),
        Switch_SOC=("Switch_SOC", "first"),
        cells=("battery", "size")
    )
    h1 = policy_table[["DQ_logvar", "cycle_life"]].dropna()
    x, y = h1.to_numpy().T
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx-rx.mean(), ry-ry.mean()
    denom = np.linalg.norm(rx)*np.linalg.norm(ry)
    rho = float(rx @ ry / denom)
    rng_perm = np.random.default_rng(seed)
    null = np.array([rx @ rng_perm.permutation(ry) / denom
                     for _ in range(permutations)])
    # 양측 검정: H0=정책 수준에서 두 변수의 독립성. 기대 방향은 음(-).
    p1 = float((1+np.sum(np.abs(null) >= abs(rho)-1e-12)) / (permutations+1))
    rng_boot = np.random.default_rng(seed+1)
    indices = rng_boot.integers(0, len(x), size=(bootstrap_samples,len(x)))
    bx, by = rankdata(x[indices],axis=1), rankdata(y[indices],axis=1)
    bx, by = bx-bx.mean(1,keepdims=True), by-by.mean(1,keepdims=True)
    den = np.sqrt(np.sum(bx**2,axis=1)*np.sum(by**2,axis=1))
    boot = np.divide(np.sum(bx*by,axis=1),den,
                     out=np.full(bootstrap_samples,np.nan),where=den>0)
    ci1 = np.nanquantile(boot,[.025,.975])

    # H2: 다른 충전 설정을 고려한 C1과 수명의 조건부 선형 관계.
    h2 = policy_table[["cycle_life","C1","C2","Switch_SOC"]].dropna()
    X = sm.add_constant(h2[["C1","C2","Switch_SOC"]])
    assert np.linalg.matrix_rank(X.to_numpy()) == X.shape[1]
    fit = sm.OLS(np.log(h2["cycle_life"]),X).fit(cov_type="HC3",use_t=True)
    beta = float(fit.params["C1"])
    p2 = float(fit.pvalues["C1"])
    ci2 = fit.conf_int().loc["C1"].to_numpy()
    reject, adjusted, _, _ = multipletests([p1,p2],alpha=.05,method="holm")
    result = pd.DataFrame([
        {"hypothesis":"H1: DQ vs life","n_policies":len(h1),
         "effect":rho,"effect_type":"Spearman rho",
         "ci95_low":ci1[0],"ci95_high":ci1[1],
         "p_raw":p1,"p_holm":adjusted[0],"reject_H0":bool(reject[0])},
        {"hypothesis":"H2: adjusted C1 vs log life","n_policies":len(h2),
         "effect":beta,"effect_type":"log-life coefficient per +1C",
         "ci95_low":ci2[0],"ci95_high":ci2[1],
         "p_raw":p2,"p_holm":adjusted[1],"reject_H0":bool(reject[1])}
    ])
    vif = pd.DataFrame({
        "term": X.columns,
        "VIF": [variance_inflation_factor(X.to_numpy(),i) for i in range(X.shape[1])]
    })
    # 파생변수는 초기 정보만 사용하고 모든 batch에 동일한 식을 적용.
    derived_rows=[]
    for name,cells in batches.items():
        for cell in cells:
            a=cell["features"]
            q10,q100=a["Qd_10"],a["Qd_100"]
            delta=cell["delta"]
            values={
                "DQ_relative_logvar":np.nan,
                "DQ_relative_loss_area":np.nan,
                "capacity_retention_100_pct":np.nan,
                "C1_to_C2_ratio":np.nan
            }
            if np.isfinite(q10) and q10>0:
                values["capacity_retention_100_pct"]=100*q100/q10
                if delta is not None:
                    relative=delta/q10
                    values["DQ_relative_logvar"]=np.log10(max(np.var(relative),1e-16))
                    order=np.argsort(cell["voltage"])
                    v=cell["voltage"][order]
                    values["DQ_relative_loss_area"]=(
                        np.trapezoid(np.maximum(-relative[order],0),v)/(v[-1]-v[0])
                    )
            if np.isfinite(a["C2"]) and a["C2"]>0:
                values["C1_to_C2_ratio"]=a["C1"]/a["C2"]
            derived_rows.append({"batch":name,"battery":cell["battery"],**values})
    derived = features.merge(pd.DataFrame(derived_rows),on=["batch","battery"],
                             validate="one_to_one")
    new_cols=["DQ_relative_logvar","DQ_relative_loss_area",
              "capacity_retention_100_pct","C1_to_C2_ratio"]
    derived_train=derived.loc[derived["batch"].eq("Batch 1")]
    # 기술 통계/상관만 확인. 새 가설 p값을 추가하여 유의한 feature를 쫓지 않음.
    feature_audit=pd.DataFrame({
        "missing":derived_train[new_cols].isna().sum(),
        "rho_with_life":derived_train[new_cols].corrwith(
            derived_train["cycle_life"],method="spearman"),
    })
    feature_correlations=derived_train[
        new_cols+["DQ_logvar","DQ_mean","DQ_min","Qd_change_10_100"]
    ].corr(method="spearman")
    return {
        "results":result,"policy_table":policy_table,
        "h1_null":null,"h1_bootstrap":boot,"h2_fit":fit,"h2_design":X,
        "vif":vif,"derived_features":derived,"derived_audit":feature_audit,
        "derived_correlations":feature_correlations,
        "h2_percent_per_1C":100*np.expm1(beta),
        "h2_percent_ci":100*np.expm1(ci2),
        "h2_residual_df":int(fit.df_resid),
    }
