# 데이터 안내

`battery_modeling_ready.parquet`는 제공된 MATLAB 원본에서 만든 셀당 1행의 초기 100사이클 특성 데이터다. 원본 대용량 `.mat`는 포함하지 않는다.

| 과제 batch | 원본 파일 | 용도 |
|---|---|---|
| Batch 1 | 2017-05-12_batchdata_updated_struct_errorcorrect.mat | 개발 37셀/18정책, Hold-out 9셀/5정책 |
| Batch 2 | 2018-02-20_batchdata_updated_struct_errorcorrect.mat | 테스트 47셀 중 39개 수명 라벨 |
| Batch 3 | 2018-04-12_batchdata_updated_struct_errorcorrect.mat | 테스트 46셀 중 44개 수명 라벨 |

139셀, 기본 피처 16개 + 파생 피처 4개. 타깃 결측 10개는 예측만 하고 평가에서 제외한다. `batch`, `battery`, `policy`, `group_id`, `battery_id`, `split`, `has_label`, `outer_cv_fold`, `source_split`, `cycle_life`는 입력 피처가 아니다.

`df.attrs`에는 피처 목록, S1/S2/S3 조합, 고정 분할, 중첩 CV 셀 ID, 계산식과 원본 SHA256이 포함되어 있다. `train`에서만 전처리를 fit하며 `valid`, `test1`, `test2` 점수로 모델을 바꾸지 않는다.

정책 Hold-out은 GroupShuffleSplit(test_size=0.2, random_state=20261001)로 고정했다. 20%는 셀 수가 아니라 정책 수 기준이다. `outer_cv_fold`는 개발 세트의 외부 CV 검증 fold이며 다른 세트에는 NA다.

## 원본을 이용한 재현

`.mat` 파일을 직접 준비하고 `BATTERY_DATA_DIR` 환경변수에 폴더를 지정한 뒤 `notebooks/01_EDA.ipynb`의 원본 재현 코드를 실행한다. `src/raw_eda.py`는 원시 Summary 전체와 cycle 10/100의 Qdlin을 읽는다. `src/hypothesis_tests.py`는 가설 검정 및 파생변수 생성 코드다.

과제의 2018-02-20 Batch 2는 원논문 메인 Batch 2(2017-06-30)와 다르다. 이 프로젝트는 논문 직접 재현이 아니다. `cycle_life`는 제공 label이며, Batch 1·3의 기록 종료와 실제 EOL 기준 일치 여부는 미확인이다.

기존 EDA에서 전체 batch를 이미 관찰했다. 따라서 Hold-out/test를 완전 미열람 데이터라고 주장하지 않는다. 포함 데이터 및 원본의 공개 재배포 조건은 원 제공처 정책을 확인해야 한다.
