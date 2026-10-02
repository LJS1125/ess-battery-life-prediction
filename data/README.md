# data

원본 데이터(.mat)는 용량이 커서(약 8.4GB) 저장소에 포함하지 않는다.

## 받는 방법
1. Kaggle [MIT-Stanford Dataset](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 데이터를 내려받는다.
2. 아래 파일을 이 폴더(`data/`)에 넣는다.

| 파일 | 용도 |
|---|---|
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 (학습) |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 (테스트) |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 (추가 테스트) |
| `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat` | 사용하지 않음 |

## 전처리 캐시
`notebooks/01_EDA.ipynb`를 처음 실행하면 `src/load_data.py`가 `.mat` 파일을 읽어 `data/processed/`에 캐시를 만든다(저장소 미포함).

출처: Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.
