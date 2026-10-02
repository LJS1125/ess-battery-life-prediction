"""DAY 1 보고서에 들어간 숫자를 데이터에서 다시 계산해 확인한다. 실행: .venv/bin/python src/verify_report_numbers.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_cells, load_summary, load_qdlin, clean_summary  # noqa: E402
from features import build_features, dq_features  # noqa: E402
from degradation import degradation_table  # noqa: E402

c = load_cells()
s = clean_summary(load_summary())
ok = c[c.status == "OK"].copy()
ok["new"] = ok.policy.str.contains("newstructure")
ok["group"] = np.where(ok.batch == "b2", np.where(ok.new, "B2-new", "B2-old"), ok.batch.str.upper())
F = build_features(ok, s).merge(ok[["cell", "group"]], on="cell")
deg = degradation_table(ok, s)
Fd = F.merge(deg[["cell", "fade_early", "fade_late", "knee_ratio"]], on="cell")


def show(title, obj):
    print(f"\n## {title}\n{obj}")


# 2장 데이터 정리
show("셀 상태", pd.crosstab(c.batch, c.status, margins=True))
b1 = c[c.batch == "b1"]
cen = b1[b1.status == "CENSORED"]
show("중도 절단 기록 끝 용량 범위", (cen.qd_end.min().round(3), cen.qd_end.max().round(3)))
show("중도 절단 기록 길이 > 정상 중앙값 수", int((cen.n_cycles > b1[b1.status == "OK"].cycle_life.median()).sum()))
show("Batch 1 최대 수명 (전체 / 정상)", (b1.cycle_life.max(), b1[b1.status == "OK"].cycle_life.max()))
show("정상 셀 수명 중앙값·범위", ok.groupby("batch").cycle_life.agg(["count", "median", "min", "max"]))

# 4.1 분포
rows = []
for b, g in ok.groupby("batch"):
    q1, q3 = g.cycle_life.quantile([.25, .75]); iqr = q3 - q1
    rows.append({"batch": b, ">1000": round((g.cycle_life > 1000).mean() * 100), "<500": round((g.cycle_life < 500).mean() * 100),
                 "low_out": int((g.cycle_life < q1 - 1.5 * iqr).sum()), "high_out": int((g.cycle_life > q3 + 1.5 * iqr).sum())})
show("장·단수명 비율(%), IQR 이상치", pd.DataFrame(rows))
show("그룹별 수명 중앙값", ok.groupby("group").cycle_life.agg(["count", "median"]))
ok["base"] = ok.policy.str.replace("-newstructure", "", regex=False)
show("6C(60%)-3C 배치별 평균 수명", ok[ok.base == "6C(60%)-3C"].groupby("batch").cycle_life.mean().round(0))
show("왜도 (전체 정상 셀) 원값 → log10", (round(ok.cycle_life.skew(), 2), round(np.log10(ok.cycle_life).skew(), 2)))
show("최단 3개 (배치별)", ok.sort_values("cycle_life").groupby("batch").head(3)[["batch", "policy", "cycle_life"]])

# 4.2 열화
cap = s[s.cycle.isin([2, 100])].pivot(index="cell", columns="cycle", values="QD")
cap["chg"] = (cap[100] / cap[2] - 1) * 100
cap = cap.join(ok.set_index("cell")[["batch"]], how="inner")
show("2→100 용량 변화 중앙값(%)", cap.groupby("batch").chg.median().round(2))
med = deg.groupby("batch")[["fade_early", "fade_late", "knee_ratio"]].median()
show("말기/초기 감소 속도 배수", (med.fade_late / med.fade_early).round(0))
show("무릎점/수명 범위·중앙값", deg.groupby("batch").knee_ratio.agg(["min", "median", "max"]).round(3))
show("corr(무릎점, 수명)", deg.groupby("batch").apply(lambda g: round(g.knee.corr(g.cycle_life), 3)))
show("corr(초기 감소 속도, 수명)", deg.groupby("batch").apply(lambda g: round(g.fade_early.corr(g.cycle_life), 2)))

# 4.3 ΔQ
lab = ["dq100_logvar", "dq100_logmin", "dq100_logmean", "dq100_skew", "dq100_kurt"]
show("ΔQ 통계 ↔ log 수명", pd.DataFrame({b: g[lab].corrwith(g.log_life) for b, g in F.groupby("batch")}).round(2))
show("그룹 내 corr(ΔQ 분산, log 수명)", F.groupby("group").apply(lambda g: round(g.dq100_logvar.corr(g.log_life), 2)))
show("ΔQ 분산 중앙값 (그룹별)", F.groupby("group").dq100_logvar.median().round(2))
early = {}
for b in ["b1", "b2", "b3"]:
    q, v, cyc, cid = load_qdlin(b)
    for n in [20, 30, 50, 100]:
        f = dq_features(q, cyc, a=n); f["cell"] = cid
        g = f.merge(F[F.batch == b][["cell", "log_life"]], on="cell")
        early[(b, n)] = round(g[f"dq{n}_logvar"].corr(g.log_life), 2)
show("ΔQ 계산 구간별 상관", pd.Series(early).unstack())

# 4.4 충전
b1f = F[F.batch == "b1"]
show("Batch 1 평균 C-rate 구간별 평균 수명", b1f.groupby(pd.cut(b1f.C_avg, [3.9, 4.4, 4.8, 5.5])).cycle_life.mean().round(0))
show("Batch 1 corr(C_avg, 수명), corr(C1, 수명), corr(C2, 수명)",
     (round(b1f.C_avg.corr(b1f.cycle_life), 2), round(b1f.C1.corr(b1f.cycle_life), 2), round(b1f.C2.corr(b1f.cycle_life), 2)))
sub = b1f[(b1f.C_avg > 4.7) & (b1f.C_avg < 4.9)]
b3f = F[F.batch == "b3"]
show("충전 시간 약 10분: corr(C_max, 수명) Batch1 부분 / Batch3", (round(sub.C_max.corr(sub.cycle_life), 2), round(b3f.C_max.corr(b3f.cycle_life), 2)))
show("Batch 3 정책별 평균 수명 (최장/최단)", b3f.groupby("policy").cycle_life.mean().round(0).sort_values().iloc[[0, -1]])
show("평균 C-rate 범위 (배치별)", F.groupby("batch").C_avg.agg(["min", "max"]).round(2))
rate = {b: {"Cavg-early": g.C_avg.corr(g.fade_early), "Cavg-late": g.C_avg.corr(g.fade_late),
            "Cmax-early": g.C_max.corr(g.fade_early), "Cmax-late": g.C_max.corr(g.fade_late)} for b, g in Fd.groupby("batch")}
show("충전 지표 ↔ 열화 속도", pd.DataFrame(rate).round(2))

# 4.5 상관
fe = Fd.assign(fade_speed=-Fd.fade_slope_2_100)
cols = ["dq100_logvar", "C_max", "chargetime_5", "fade_speed", "ir_min", "tavg_mean"]
show("주요 피처 ↔ log 수명", pd.DataFrame({b: g[cols].corrwith(g.log_life) for b, g in fe.groupby("batch")}).round(2))
cm = b1f[["dq100_logvar", "dq100_logmin", "dq100_logmean", "chargetime_5", "C_avg"]].corr().round(2)
show("Batch 1 피처 간 상관 (중복)", cm)


def vif(df):
    X = ((df - df.mean()) / df.std()).assign(const=1.0)
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1] - 1)], index=df.columns).round(1)


show("VIF (ΔQ 대표 1개)", vif(b1f[["dq100_logvar", "qd_100_minus_2", "chargetime_5", "C_max"]]))
show("IR 결측 셀 수 (배치별)", F[F.ir_2.isna()].groupby("batch").size())

# 4.6 배치 비교
k, c0 = np.polyfit(b1f.dq100_logvar, b1f.log_life, 1)
F["resid"] = F.log_life - (k * F.dq100_logvar + c0)
fits = {g: np.polyfit(d.dq100_logvar, d.log_life, 1)[0].round(2) for g, d in F.groupby("group")}
show("그룹별 기울기", fits)
show("Batch 1 관계식 대비 실제 수명 중앙값(%)", F.groupby("group").resid.median().pipe(lambda x: ((10 ** x - 1) * 100).round(1)))
old = F[F.group == "B2-old"]
pred = 10 ** (k * old.dq100_logvar + c0)
show("B2 기존 방식: 과대추정 비율, 중앙값 차이(사이클), 중앙값 %",
     (round((pred > old.cycle_life).mean(), 2), round(float(np.median(pred - old.cycle_life))), round(float(np.median(pred / old.cycle_life - 1) * 100), 1)))
show("MAPE 9.1% × 773 (사이클, 개월@1/일)", (round(0.091 * 773), round(0.091 * 773 / 30.4, 1)))
