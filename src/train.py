"""
DAY 2 학습·평가 파이프라인

    python src/train.py          # 전체 실행: 피처 → 분할 → 튜닝 → 선택 → 테스트 → 결과 저장

흐름 (DAY 1 보고서 5.1~5.3을 그대로 구현)
1. 데이터: Batch 1 정상 36셀 = 학습, Batch 2 39셀 = 테스트, Batch 3 44셀 = 추가 테스트
2. 분할: Batch 1을 충전 정책 그룹 단위로 Train 약 75% / Valid(Hold-out) 약 25%로 나눈다 (seed 42 → 28셀 / 8셀)
         → 같은 정책의 쌍둥이 셀이 양쪽에 갈라지지 않음 (노션이 경고한 누수 차단)
3. 튜닝: Train 셀 안에서 GroupKFold(정책 단위) + GridSearchCV
4. 선택: ① Valid MAPE ② 과대예측 비율 ③ CV–Valid 차이 ④ 단순한 모델 (Batch 2·3은 보지 않음)
5. 테스트: 선택한 모델을 Batch 1 전체(36셀)로 다시 학습 → Batch 2, Batch 3 각 1회 평가
- 타깃은 log10(cycle_life)로 학습하고, MAPE는 사이클 수로 되돌려 계산 (원논문과 같은 단위)
- 표준화·결측 대체는 Pipeline 안에서 학습 데이터로만 맞춤 (테스트 정보 누수 차단)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_SETS, build_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 42
VALID_SIZE = 0.25
N_FOLDS = 5
TARGET_MAPE = 9.1          # 원논문 회귀 테스트 오차

warnings.filterwarnings("ignore", category=UserWarning)


# ── 모델 후보 (DAY 1 보고서 5.2) ────────────────────────────────────
def _pipe(reg, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]   # Batch 2 일부 셀의 내부저항 결측
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("reg", reg))
    # 타깃을 log10으로 바꿔 학습하고, 예측은 다시 사이클 수로 되돌림
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log10,
                                      inverse_func=lambda z: 10 ** z, check_inverse=False)


P = "regressor__reg__"
CANDIDATES = {
    # 이름: (모델, 하이퍼파라미터 탐색 범위, 사용할 피처 세트)
    "평균 예측": (_pipe(DummyRegressor()), {}, ["A"]),
    "선형 회귀": (_pipe(LinearRegression()), {}, ["A"]),
    "ElasticNet": (_pipe(ElasticNet(max_iter=50_000)),
                   {P + "alpha": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                    P + "l1_ratio": [0.1, 0.5, 0.9, 1.0]}, ["B", "C", "D"]),
    "랜덤포레스트": (_pipe(RandomForestRegressor(random_state=SEED), scale=False),
                {P + "n_estimators": [300],
                 P + "max_depth": [2, 3, None],
                 P + "min_samples_leaf": [1, 3, 5]}, ["A", "B", "C"]),
    "그래디언트부스팅": (_pipe(GradientBoostingRegressor(random_state=SEED), scale=False),
                  {P + "n_estimators": [100, 300],
                   P + "learning_rate": [0.03, 0.1],
                   P + "max_depth": [1, 2, 3]}, ["A", "B", "C"]),
}
# 같은 성능이면 단순한 모델을 고르기 위한 복잡도 순서 (선택 기준 ④)
COMPLEXITY = {"평균 예측": 0, "선형 회귀": 1, "ElasticNet": 2, "랜덤포레스트": 3, "그래디언트부스팅": 3}


# ── 지표 ───────────────────────────────────────────────────────────
def mape(y, p):
    return float(np.mean(np.abs(p - y) / y) * 100)


def over_ratio(y, p):
    """실제보다 길게 예측한 셀 비율(%) — 교체 시점을 놓치는 위험한 방향"""
    return float(np.mean(p > y) * 100)


def evaluate(y, p):
    return {"MAPE": mape(y, p), "과대예측비율": over_ratio(y, p),
            "중앙값오차": float(np.median((p - y) / y) * 100), "n": int(len(y))}


# ── 분할 ───────────────────────────────────────────────────────────
def split_batch1(b1, seed=SEED):
    """충전 정책 그룹 단위 Hold-out. 반환: (train, valid)"""
    gss = GroupShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=seed)
    tr, va = next(gss.split(b1, groups=b1.group))
    return b1.iloc[tr], b1.iloc[va]


def tune(name, fs, train):
    """주어진 셀 안에서 정책 단위 GroupKFold로 하이퍼파라미터 탐색. 반환: (fitted search, CV MAPE)"""
    model, grid, _ = CANDIDATES[name]
    cols = FEATURE_SETS[fs]
    n_groups = train.group.nunique()
    gs = GridSearchCV(model, grid or {}, cv=GroupKFold(n_splits=min(N_FOLDS, n_groups)),
                      scoring="neg_mean_absolute_percentage_error", n_jobs=-1)
    gs.fit(train[cols], train.cycle_life, groups=train.group)
    return gs, -gs.best_score_ * 100


def compare_models(train, valid):
    """후보 × 피처 세트 비교표 (Batch 1 안에서만)"""
    rows, fitted = [], {}
    for name, (_, _, sets) in CANDIDATES.items():
        for fs in sets:
            gs, cv = tune(name, fs, train)
            cols = FEATURE_SETS[fs]
            pv = gs.predict(valid[cols])
            ev = evaluate(valid.cycle_life.values, pv)
            rows.append({"모델": name, "피처세트": fs, "피처수": len(cols),
                         "Train_CV_MAPE": cv, "Valid_MAPE": ev["MAPE"],
                         "Valid_과대예측비율": ev["과대예측비율"],
                         "Gap_Train_Valid": ev["MAPE"] - cv,
                         "최적파라미터": json.dumps({k.replace(P, ""): v for k, v in gs.best_params_.items()},
                                              ensure_ascii=False)})
            fitted[(name, fs)] = gs
    return pd.DataFrame(rows), fitted


def select(table, tol=1.0):
    """
    DAY 1 선택 규칙. Valid MAPE가 1등과 tol(%p) 이내인 후보 중에서
    과대예측 비율 → |CV–Valid 차이| → 단순함 순으로 고른다.
    (Valid가 8셀뿐이라 0.x%p 차이는 우연일 수 있어 동률 범위를 둠)
    """
    t = table[table.모델 != "평균 예측"].copy()
    best = t.Valid_MAPE.min()
    near = t[t.Valid_MAPE <= best + tol].copy()
    near["abs_gap"] = near.Gap_Train_Valid.abs()
    near["복잡도"] = near.모델.map(COMPLEXITY) * 100 + near.피처수
    return near.sort_values(["Valid_과대예측비율", "abs_gap", "복잡도"]).iloc[0]


def repeated_holdout(b1, combos, n=30):
    """Valid 셀이 8~9개뿐이라 우연에 흔들리므로 분할 seed를 바꿔 n번 반복한 Valid MAPE (참고용)"""
    rows = []
    for seed in range(n):
        tr, va = split_batch1(b1, seed)
        for name, fs in combos:
            gs, _ = tune(name, fs, tr)
            rows.append({"seed": seed, "모델": name, "피처세트": fs,
                         "Valid_MAPE": mape(va.cycle_life.values, gs.predict(va[FEATURE_SETS[fs]]))})
    return pd.DataFrame(rows)


def final_fit(name, fs, b1):
    """선택한 모델을 Batch 1 전체(36셀)로 다시 튜닝·학습"""
    gs, cv = tune(name, fs, b1)
    return gs.best_estimator_, cv


def predict_table(model, fs, df):
    out = df[["batch", "cell", "policy", "subgroup", "status", "cycle_life", "min_life"]].copy()
    out["pred"] = model.predict(df[FEATURE_SETS[fs]])
    out["err_pct"] = (out.pred - out.cycle_life) / out.cycle_life * 100
    return out


def performance_table(cv, valid_mape, pred, n_train, n_valid):
    """노션 Reporting format (Regression + Batch 3)"""
    t2 = mape(*pred.query("batch=='b2' and status=='OK'")[["cycle_life", "pred"]].values.T)
    t3 = mape(*pred.query("batch=='b3' and status=='OK'")[["cycle_life", "pred"]].values.T)
    rows = [
        ("Train (Batch 1 CV)", cv, f"Train {n_train}셀, 정책 단위 GroupKFold 평균"),
        ("Valid (Batch 1 Hold-out)", valid_mape, f"정책 단위 Hold-out {n_valid}셀"),
        ("Test (Batch 2)", t2, "Batch 1 전체로 재학습 후 1회 평가"),
        ("Gap (Train-Valid)", valid_mape - cv, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", t2 - valid_mape, "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", t2 - TARGET_MAPE, "Target : 원논문 9.1%"),
        ("Test (Batch 3)", t3, "추가 검증"),
        ("Gap (Batch2-Batch3)", t2 - t3, "Test 성능 간 비교"),
        ("Gap (Target-Test, Batch 3)", t3 - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def run():
    RESULTS.mkdir(exist_ok=True)
    df = build_dataset()
    ok = df[df.status == "OK"]
    b1 = ok[ok.batch == "b1"].reset_index(drop=True)
    train, valid = split_batch1(b1)
    print(f"Batch 1: train {len(train)}셀({train.group.nunique()}정책) / valid {len(valid)}셀({valid.group.nunique()}정책)")
    assert not set(train.group) & set(valid.group), "정책이 train/valid 양쪽에 있음"

    # 1) 후보 비교 (Batch 1 안에서만)
    table, fitted = compare_models(train, valid)
    table.to_csv(RESULTS / "model_comparison.csv", index=False, encoding="utf-8-sig")
    print(table.drop(columns="최적파라미터").round(2).to_string(index=False))

    # 2) 선택
    best = select(table)
    name, fs = best.모델, best.피처세트
    print(f"\n선택: {name} / 피처 세트 {fs}  (Valid MAPE {best.Valid_MAPE:.2f}%)")

    # 3) 최종 학습 (Batch 1 전체) → 테스트 1회
    model, cv_all = final_fit(name, fs, b1)
    pred = predict_table(model, fs, df)
    pred.to_csv(RESULTS / "predictions.csv", index=False, encoding="utf-8-sig")
    perf = performance_table(best.Train_CV_MAPE, best.Valid_MAPE, pred, len(train), len(valid))
    perf.to_csv(RESULTS / "model_performance.csv", index=False, encoding="utf-8-sig")
    print("\n" + perf.round(2).to_string(index=False))

    meta = {"model": name, "feature_set": fs, "features": FEATURE_SETS[fs],
            "params": json.loads(best.최적파라미터), "cv_mape_batch1_all": cv_all,
            "train_cells": sorted(train.cell), "valid_cells": sorted(valid.cell)}
    (RESULTS / "final_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return df, train, valid, table, fitted, model, pred, perf


if __name__ == "__main__":
    run()
