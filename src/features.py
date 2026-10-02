"""
피처 생성 — 반드시 1~100사이클 정보만 사용 (예측 시점 = 100사이클)

피처 그룹
- 실험실 피처(lab): ΔQ(V) 곡선 통계. 일정한 조건의 완전 방전 곡선이 있어야 계산 가능
- 현장 피처(field): BMS가 평소에 기록하는 값(용량, 내부저항, 온도, 충전 시간)
- 정책 피처(policy): 충전 방식 설계값 (사전에 알 수 있는 정보)
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

from load_data import load_qdlin

PRED_CYCLE = 100


def parse_policy(policy):
    """
    '5.4C(40%)-3.6C'  → C1=5.4, Q1=40, C2=3.6
    '4C(80%)-4C-newstructure' → C1=4, Q1=80, C2=4
    0→80% 평균 C-rate: 0.8 / (Q1/C1 + (0.8−Q1)/C2)   (Q1은 비율)
    """
    m = re.match(r"\s*([\d.]+)C\((\d+)%\)-([\d.]+)C", policy)
    if not m:
        return dict(C1=np.nan, Q1=np.nan, C2=np.nan, C_avg=np.nan, C_max=np.nan)
    c1, q1, c2 = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
    t = q1 / c1 + max(0.8 - q1, 0) / c2           # 0→80% 충전 시간 (시간 단위 ×1)
    return dict(C1=c1, Q1=q1 * 100, C2=c2, C_avg=0.8 / t, C_max=max(c1, c2))


def dq_features(qdlin, cycles, a=PRED_CYCLE, b=10):
    """ΔQ(V) = Q_a(V) − Q_b(V) 의 통계값 (셀별)"""
    dq = qdlin[:, cycles.index(a)] - qdlin[:, cycles.index(b)]
    return pd.DataFrame({
        f"dq{a}_logvar": np.log10(np.var(dq, axis=1)),
        f"dq{a}_logmin": np.log10(np.abs(dq.min(axis=1))),
        f"dq{a}_logmean": np.log10(np.abs(dq.mean(axis=1))),
        f"dq{a}_skew": stats.skew(dq, axis=1),
        f"dq{a}_kurt": stats.kurtosis(dq, axis=1),
    })


def summary_features(summary, max_cycle=PRED_CYCLE):
    """사이클 요약 기반 현장 피처 (2~max_cycle사이클)"""
    s = summary[(summary.cycle >= 2) & (summary.cycle <= max_cycle)]
    rows = []
    for cell, d in s.groupby("cell"):
        d = d.sort_values("cycle")
        cyc, qd = d.cycle.values, d.QD.values

        def slope(lo, hi):
            m = (cyc >= lo) & (cyc <= hi)
            return np.polyfit(cyc[m], qd[m], 1)[0] if m.sum() >= 3 else np.nan

        ir = d.IR.values
        ir_valid = ir[ir > 0]
        rows.append({
            "cell": cell,
            "qd_2": qd[0],
            "qd_max_minus_2": qd.max() - qd[0],
            "qd_100_minus_2": qd[-1] - qd[0],
            "fade_slope_2_100": slope(2, max_cycle),
            "fade_slope_91_100": slope(max_cycle - 9, max_cycle),
            "ir_2": ir_valid[0] if ir_valid.size else np.nan,
            "ir_min": ir_valid.min() if ir_valid.size else np.nan,
            "ir_100_minus_2": (ir_valid[-1] - ir_valid[0]) if ir_valid.size else np.nan,
            "tavg_mean": d.Tavg.mean(),
            "tmax_max": d.Tmax.max(),
            "chargetime_5": d.chargetime.iloc[:5].mean(),
        })
    return pd.DataFrame(rows)


FEATURE_GROUPS = {
    "lab": ["dq100_logvar", "dq100_logmin", "dq100_logmean", "dq100_skew", "dq100_kurt"],
    "field": ["qd_2", "qd_max_minus_2", "qd_100_minus_2", "fade_slope_2_100", "fade_slope_91_100",
              "ir_2", "ir_min", "ir_100_minus_2", "tavg_mean", "tmax_max", "chargetime_5"],
    "policy": ["C1", "Q1", "C2", "C_avg", "C_max"],
}


def build_features(cells, summary):
    """cells(정상 셀 등) × 전체 피처 표. summary는 clean_summary 결과."""
    frames = []
    for b in sorted(cells.batch.unique()):
        q, v, cyc, cell_ids = load_qdlin(b)
        f = dq_features(q, cyc)
        f.insert(0, "cell", cell_ids)
        frames.append(f)
    dq = pd.concat(frames, ignore_index=True)
    pol = pd.DataFrame([{"cell": r.cell, **parse_policy(r.policy)} for r in cells.itertuples()])
    out = (cells[["batch", "cell", "policy", "cycle_life"]]
           .merge(dq, on="cell", how="left")
           .merge(summary_features(summary), on="cell", how="left")
           .merge(pol, on="cell", how="left"))
    out["log_life"] = np.log10(out.cycle_life)
    return out


# ── DAY 2: 피처 세트 (DAY 1 보고서 5.1) ─────────────────────────────
# A: ΔQ 단일 (원논문 재현, 기준)
# B: A + 충전 정책 (운영 변수 추가)
# C: 전체 후보 + 규제 (모델이 스스로 고르게)
# D: 현장 피처만 (ΔQ 없이 BMS 기록으로 어디까지 되는지)
FEATURE_SETS = {
    "A": ["dq100_logvar"],
    "B": ["dq100_logvar", "C_avg", "C_max"],
    "C": ["dq100_logvar", "dq100_logmin", "dq100_logmean", "dq100_skew", "dq100_kurt",
          "C_avg", "C_max",
          "qd_2", "fade_slope_2_100", "chargetime_5", "ir_min", "tavg_mean"],
    "D": ["qd_2", "qd_100_minus_2", "fade_slope_2_100", "fade_slope_91_100",
          "ir_min", "ir_100_minus_2", "tavg_mean", "chargetime_5"],
}


def build_dataset(pred_cycle=PRED_CYCLE):
    """
    학습·평가에 쓰는 셀 전체의 피처 표.
    - status OK: 학습(B1)·테스트(B2, B3)
    - status CENSORED(B1 10셀) + B3 NO_LABEL 2셀(2,189·2,237사이클까지 0.88Ah 미도달):
      정답은 없지만 '최소 이만큼은 살았다'(min_life = 기록된 사이클 수)를 안다 → 최소 수명 점검용
    """
    from load_data import build_cache, load_cells, load_summary, clean_summary
    build_cache()                           # 캐시가 없으면 .mat에서 생성 (있으면 건너뜀)
    cells = load_cells()
    b3_censored = (cells.batch == "b3") & (cells.status == "NO_LABEL") & (cells.qd_end > 0.89)
    cells.loc[b3_censored, "status"] = "CENSORED"
    use = cells[cells.status.isin(["OK", "CENSORED"])].copy()
    df = build_features(use, clean_summary(load_summary()))
    if pred_cycle != PRED_CYCLE:            # 50사이클 등 예측 시점을 앞당긴 버전
        frames = []
        for b in sorted(use.batch.unique()):
            q, v, cyc, ids = load_qdlin(b)
            f = dq_features(q, cyc, a=pred_cycle)
            f.insert(0, "cell", ids)
            frames.append(f)
        df = df.merge(pd.concat(frames), on="cell", how="left")
    df = df.merge(use[["cell", "status", "n_cycles"]], on="cell")
    df["group"] = df.policy.str.replace("-newstructure", "", regex=False)
    df["subgroup"] = np.where(df.policy.str.contains("newstructure"), "newstructure", "기존 방식")
    df["min_life"] = np.where(df.status == "CENSORED", df.n_cycles, np.nan)
    return df
