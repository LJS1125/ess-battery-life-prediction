"""열화 곡선 분석: 무릎점(knee point), 열화 속도"""
import numpy as np
import pandas as pd


def knee_point(cycles, qd, window=9):
    """
    무릎점 = 용량이 '천천히 감소'하다가 '급격히 감소'로 바뀌는 지점.
    방법: 이동 중앙값으로 매끄럽게 한 곡선에서, 시작점과 끝점을 잇는 직선과
    가장 멀리 떨어진 점 (Kneedle 방식의 단순 버전).
    """
    q = pd.Series(qd).rolling(window, center=True, min_periods=1).median().values
    x, y = np.asarray(cycles, float), q
    x0, y0, x1, y1 = x[0], y[0], x[-1], y[-1]
    # 각 점과 직선 사이 거리 (곡선이 직선 위쪽에 있는 볼록한 형태)
    dist = ((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / np.hypot(y1 - y0, x1 - x0)
    i = int(np.argmax(-dist)) if np.max(-dist) > 0 else int(np.argmax(dist))
    return x[i]


def fade_rate(cycles, qd, start, end):
    """구간 [start, end] 사이클의 용량 감소 기울기 (Ah/100사이클, 양수 = 감소)"""
    m = (cycles >= start) & (cycles <= end)
    if m.sum() < 5:
        return np.nan
    slope = np.polyfit(cycles[m], qd[m], 1)[0]
    return -slope * 100


def degradation_table(cells, summary):
    rows = []
    for _, r in cells.iterrows():
        d = summary[summary.cell == r.cell]
        cyc, qd = d.cycle.values, d.QD.values
        if len(cyc) < 100:
            continue
        life = r.cycle_life
        knee = knee_point(cyc, qd)
        rows.append({
            "cell": r.cell, "batch": r.batch, "cycle_life": life,
            "knee": knee,
            "knee_ratio": knee / life,
            "fade_early": fade_rate(cyc, qd, 2, 100),           # 초기 100사이클
            "fade_late": fade_rate(cyc, qd, life - 100, life),  # 수명 끝 직전 100사이클
        })
    t = pd.DataFrame(rows)
    t["accel"] = t.fade_late / t.fade_early.clip(lower=1e-4)
    return t
