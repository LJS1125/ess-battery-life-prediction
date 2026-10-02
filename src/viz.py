"""장표용 그래프 공통 스타일"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"

BATCH_COLOR = {"b1": "#2a78d6", "b2": "#e8590c", "b3": "#1f9d55"}
BATCH_LABEL = {"b1": "Batch 1 (학습)", "b2": "Batch 2 (테스트)", "b3": "Batch 3 (추가 테스트)"}
STATUS_COLOR = {"OK": "#8a8f98", "CENSORED": "#d6336c", "NO_LABEL": "#f0a020",
                "TOO_SHORT": "#7048e8", "ANOMALY": "#000000"}
STATUS_LABEL = {"OK": "정상", "CENSORED": "수명 끝 전 실험 종료",
                "NO_LABEL": "정답 없음", "TOO_SHORT": "기록 100사이클 미만", "ANOMALY": "비정상 값"}


def setup():
    plt.rcParams.update({
        "font.family": "AppleGothic",
        "axes.unicode_minus": False,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
