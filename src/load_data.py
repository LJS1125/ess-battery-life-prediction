"""
.mat(MATLAB v7.3) 배터리 데이터를 읽어 분석하기 쉬운 형태로 캐시한다.

- 배치 파일 하나 = 배터리 셀 여러 개
- 셀마다: cycle_life(정답), policy(충전 방식), summary(사이클별 요약), cycles(사이클 내부 곡선)
- 인덱스 규칙: summary/cycles의 i번째 값 = (i+1)번째 사이클

산출물 (data/processed/)
- cells.parquet   : 셀 1개 = 1행 (배치, 정책, 수명, 기록 길이, 상태)
- summary.parquet : 셀 × 사이클 1행 (QD, QC, IR, 온도, 충전 시간)
- qdlin_<batch>.npz : 선택한 사이클들의 Qdlin 곡선 (ΔQ(V) 계산용) + Vdlin(전압 축)
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"

BATCH_FILES = {
    "b1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "b2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "b3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
BATCH_LABEL = {"b1": "Batch 1", "b2": "Batch 2", "b3": "Batch 3"}

# ΔQ(V)와 조기 예측 분석에 쓸 사이클 번호
QDLIN_CYCLES = [2, 5, 10, 20, 30, 50, 100]
SUMMARY_FIELDS = {
    "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
    "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime",
}

# 셀 상태 분류 기준 (근거는 DECISIONS.md)
EOL_AH = 0.88          # 공칭 1.1Ah의 80% = 수명 끝
N_REQUIRED = 100       # 피처 계산에 필요한 최소 사이클 수
QD_MAX_VALID = 1.3     # 공칭 1.1Ah 대비 비정상적으로 큰 용량
CENSOR_AH = 0.89       # 기록 끝 용량이 이보다 크면 수명 끝 미도달


def _ref_array(f, ref):
    return np.array(f[ref]).ravel()


def _read_string(f, ref):
    """MATLAB char 배열 → 파이썬 문자열"""
    arr = np.array(f[ref]).ravel()
    return "".join(chr(int(c)) for c in arr)


def _classify(row):
    if np.isnan(row["cycle_life"]):
        return "NO_LABEL"
    if row["n_cycles"] < N_REQUIRED:
        return "TOO_SHORT"
    # 피처 구간(1~100사이클)에 비정상 값이 있으면 피처를 믿을 수 없음
    # 한두 사이클 순간 튐은 보정(spike 처리)하고 셀은 유지, 여러 사이클이 이상하면 제외
    if row["n_spike_early"] > 2:
        return "ANOMALY"
    # 기록 끝 용량이 0.88Ah보다 확실히 높으면 = 수명 끝 전에 실험 종료
    # (0.885 근처는 반올림 수준이라 EOL 도달로 봄)
    if row["qd_end"] > CENSOR_AH:
        return "CENSORED"
    return "OK"


def extract_batch(key):
    path = DATA_DIR / BATCH_FILES[key]
    f = h5py.File(path, "r")
    b = f["batch"]
    n_cells = b["summary"].shape[0]

    vdlin = None
    cell_rows, summary_frames = [], []
    qdlin = np.full((n_cells, len(QDLIN_CYCLES), 1000), np.nan)

    for i in range(n_cells):
        cl = _ref_array(f, b["cycle_life"][i, 0])
        cycle_life = float(cl[0]) if cl.size else np.nan
        policy = _read_string(f, b["policy_readable"][i, 0])

        s = f[b["summary"][i, 0]]
        summ = {new: np.array(s[old]).ravel() for old, new in SUMMARY_FIELDS.items()}
        n = len(summ["QD"])
        sdf = pd.DataFrame(summ)
        sdf.insert(0, "cycle", np.arange(1, n + 1))
        sdf.insert(0, "cell", f"{key}_c{i:02d}")
        sdf.insert(0, "batch", key)
        summary_frames.append(sdf)

        c = f[b["cycles"][i, 0]]
        n_cyc_rec = c["Qdlin"].shape[0]
        for j, cyc in enumerate(QDLIN_CYCLES):
            if cyc - 1 < n_cyc_rec:
                q = _ref_array(f, c["Qdlin"][cyc - 1, 0])
                if q.size == 1000:
                    qdlin[i, j] = q
        if vdlin is None:
            vdlin = _ref_array(f, b["Vdlin"][i, 0])

        qd = summ["QD"]
        qd_valid = qd[qd > 0]
        cell_rows.append({
            "batch": key,
            "cell": f"{key}_c{i:02d}",
            "policy": policy,
            "cycle_life": cycle_life,
            "n_cycles": n,
            "qd_first": float(qd_valid[0]) if qd_valid.size else np.nan,
            # 마지막 5사이클 중앙값: 끝부분 순간 튐(spike)에 덜 민감하게
            "qd_end": float(np.median(qd[-5:])),
            "qd_max_early": float(qd[1:min(n, N_REQUIRED)].max()) if n > 1 else np.nan,
            "n_spike_early": int((qd[:min(n, N_REQUIRED)] > QD_MAX_VALID).sum()),
            "qd_max_all": float(qd.max()),
            "slowcycle": "SLOWCYCLE" in policy.upper(),
            # 100사이클 이후 순간 튐(측정 오류) 여부 — 정답에는 영향 없음, 기록만
            "spike_late": bool(qd[N_REQUIRED:].max() > QD_MAX_VALID) if n > N_REQUIRED else False,
        })

    cells = pd.DataFrame(cell_rows)
    cells["status"] = cells.apply(_classify, axis=1)
    summary = pd.concat(summary_frames, ignore_index=True)
    return cells, summary, qdlin, vdlin


def build_cache(force=False):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cells_path, summ_path = PROC_DIR / "cells.parquet", PROC_DIR / "summary.parquet"
    if cells_path.exists() and summ_path.exists() and not force:
        return
    all_cells, all_summ = [], []
    for key in BATCH_FILES:
        print(f"[{BATCH_LABEL[key]}] 읽는 중...", flush=True)
        cells, summary, qdlin, vdlin = extract_batch(key)
        all_cells.append(cells)
        all_summ.append(summary)
        np.savez_compressed(PROC_DIR / f"qdlin_{key}.npz",
                            qdlin=qdlin, vdlin=vdlin,
                            cycles=np.array(QDLIN_CYCLES), cells=cells["cell"].values)
    pd.concat(all_cells, ignore_index=True).to_parquet(cells_path)
    pd.concat(all_summ, ignore_index=True).to_parquet(summ_path)
    print("캐시 저장 완료:", PROC_DIR)


def load_cells():
    return pd.read_parquet(PROC_DIR / "cells.parquet")


def load_summary():
    return pd.read_parquet(PROC_DIR / "summary.parquet")


def load_qdlin(key):
    """반환: qdlin[셀, 사이클(QDLIN_CYCLES 순서), 1000], vdlin[1000], cycles, cells"""
    z = np.load(PROC_DIR / f"qdlin_{key}.npz", allow_pickle=True)
    return z["qdlin"], z["vdlin"], list(z["cycles"]), list(z["cells"])


if __name__ == "__main__":
    build_cache(force=True)


def clean_summary(summary, max_cycle=None):
    """
    사이클 요약 정리
    - 1번 사이클 중 기록이 0인 행 제거 (Batch 1)
    - 순간 튐(QD > 1.3Ah) 값은 앞뒤 사이클로 보간
    - max_cycle을 주면 그 사이클까지만 남김 (예: 100 → 예측 시점까지의 정보만)
    """
    s = summary.copy()
    s = s[~((s["cycle"] == 1) & (s["QD"] <= 0))]
    s.loc[s["QD"] > QD_MAX_VALID, "QD"] = np.nan
    s["QD"] = s.groupby("cell")["QD"].transform(lambda x: x.interpolate(limit_direction="both"))
    if max_cycle is not None:
        s = s[s["cycle"] <= max_cycle]
    return s.reset_index(drop=True)
