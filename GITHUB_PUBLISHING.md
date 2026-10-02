# GitHub 공개 업로드 규정 및 체크리스트

본 문서는 프로젝트의 공개 업로드를 위한 운영 기준이다. 코드·데이터의 이용 권한을 새로 부여하는 라이선스 문서가 아니다. 공개 소스 대상 저장소는 [sumin-990416/ess-battery-project](https://github.com/sumin-990416/ess-battery-project)이며, 사용자의 저장소 생성·업로드 요청에 따라 개인 작업 폴더와 별도의 공개용 소스 사본을 사용한다. 데이터·모델의 재배포 및 코드 라이선스는 아직 확정하지 않았으므로 이후 추가 공개 전 별도 확인이 필요하다.

## 1. 업로드 범위

| 구분 | 기본 정책 | 주의사항 |
| --- | --- | --- |
| 직접 작성한 코드·README·테스트 | 검토 후 업로드 | 출처와 제3자 코드 이용 조건 확인 |
| `.gitignore`, `.env.example`, requirements | 업로드 | 예시에도 실제 키·비밀번호·개인 경로를 넣지 않음 |
| `.env`, 개인 설정·인증 키 | 업로드 금지 | 실제 값은 로컬 또는 비밀 관리 기능에 보관 |
| 원본 `.mat`, 파생 Parquet, 셀별 예측 CSV | 기본 제외 | 공개 데이터라고 재배포가 자동 허용되는 것은 아님 |
| 모델 joblib/pickle, Optuna DB·로그 | 기본 제외 | 배포 권한과 학습 데이터 이용 조건을 별도 확인 |
| 집계 성능표·직접 작성한 그래프 | 검토 후 업로드 후보 | 원자료 이용 조건·식별정보·저작권은 별도 검토 |
| 노트북 | 출력 제거 후 업로드 | 코드·Markdown·경로·링크는 수동 점검도 필요 |
| 논문 PDF·과제 자료 PDF·개인용 ZIP | 기본 제외 | 원문 링크와 출처를 제공하고 무단 재배포하지 않음 |

현재 `.gitignore`는 하위 폴더 모델까지 포함하여 데이터·모델·DB·로그·비밀정보를 제외한다. 학습용 개인 ZIP에는 재현을 위해 데이터와 모델이 남아 있으므로 **개인용 ZIP을 그대로 GitHub에 올리지 않는다.** 공개용 패키지를 별도로 사용한다.

## 2. 환경 설정

`.env.example`은 공개 가능한 템플릿이고 `.env`는 로컬 설정이다. 현재 모델링에 API 키는 필요하지 않으며 원시 데이터 경로만 설정한다. 다음 명령은 기존 `.env`가 없을 때만 사용한다.

```bash
cp -n .env.example .env
pip install -r requirements.txt
```

`.env`의 `BATTERY_DATA_DIR`를 자신의 원시 archive 경로로 수정한다. `src.prepare`와 `src.raw_eda`는 프로젝트 루트의 `.env`를 읽으며, 이미 설정한 프로세스 환경변수를 덮어쓰지 않는다. `src.prepare --raw`는 환경 설정보다 우선한다. 이 설정은 원본 EDA/피처 재구성용이고, 이미 준비한 Parquet의 위치나 모든 모델링 명령을 자동으로 바꾸지는 않는다.

```bash
python -m src.prepare --output data/rebuilt.parquet
```

공개용 패키지에서는 데이터가 빠져 있으므로 이 경로를 설정하고 허용된 원자료를 별도로 확보해야 한다. 공개 clone만으로 저장된 모델 예측까지 바로 실행되는 패키지라고 설명하지 않는다.

## 3. GitHub용 패키지 생성

```bash
python scripts/make_public_bundle.py --out dist/ess_battery_github.zip
```

이 도구는 `.gitignore`를 실제 Git 규칙으로 적용하고 노트북의 실행 출력·실행 횟수·추가 메타데이터·첨부물을 제외한다. 원본 노트북은 변경하지 않는다. 일부 토큰/개인 키 패턴과 파일 크기를 검사하며 탐지한 값 자체는 출력하지 않는다. 점검을 위해 임시 Git 저장소만 사용하고 프로젝트 저장소를 초기화하거나 외부로 전송하지 않는다.

**한계:** 이 검사는 완전한 비밀정보 탐지기가 아니며 저작권·라이선스 확인을 대신하지 않는다. 코드와 Markdown에 직접 적힌 개인정보, 원자료에서 파생된 집계/그래프의 이용 조건, Git 이력은 별도 확인한다. 공개용 ZIP을 풀어 확인한 소스 폴더에서 업로드를 준비한다. ZIP 파일 자체는 저장소에 커밋하지 않는다.

## 4. 데이터와 라이선스

- 데이터와 공식 코드의 출처: [원논문](https://doi.org/10.1038/s41560-019-0356-8), [공식 저장소](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation), [원자료 제공처](https://data.matr.io/1/).
- 공식 저장소는 데이터 처리 코드와 모델링 코드의 접근 조건을 구분한다. 모델링 코드는 저자 요청 및 학술 라이선스 안내 대상이므로 내려받은 코드를 자유 재배포 가능한 것으로 가정하지 않는다.
- 공개 데이터라는 사실과 원본·파생 데이터·학습 모델의 재배포 권한은 동일하지 않다. 각 이용 조건을 확인하고 확인 전에는 제외 규칙을 유지한다.
- 본 프로젝트에 MIT 등의 코드 라이선스는 임의로 추가하지 않았다. 직접 작성한 코드의 공개 라이선스 선택은 권리자와 팀/교육기관 조건을 확인한 뒤 별도로 진행한다. 코드 라이선스와 데이터/논문/과제 자료 권한을 구분한다.
- 프로젝트에 LICENSE가 없다고 다른 사람이 자유롭게 재사용할 수 있다는 뜻은 아니다. [GitHub 라이선스 안내](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).
- 소속·반·번호·이름은 사용자가 README에 포함하도록 요청한 정보다. 공개 저장소에서는 이 정보가 공개되므로 업로드 전 공개 여부를 다시 확인한다.

## 5. 커밋·push 전 확인

- [ ] `.env`, API 키·인증서·비밀번호·개인 경로가 소스/노트북/출력에 없는지 확인한다.
- [ ] 원자료·파생 데이터·모델·셀별 예측·PDF·ZIP이 추적 대상에 포함되지 않는지 확인한다.
- [ ] 노트북 실행 출력 제거를 확인하고 Markdown과 셀 코드도 읽어 본다.
- [ ] 데이터 및 제3자 코드/그래프의 이용·재배포 조건과 출처를 확인한다.
- [ ] README의 이름·소속·번호 공개 여부를 확인한다.
- [ ] 타깃 라벨 미검증, 과제/논문 데이터 차이, 테스트 사전 열람 등 한계를 숨기지 않는다.
- [ ] 코드 실행에 필요한 데이터 확보 절차와 제외된 산출물을 README에 설명한다.
- [ ] 저장소 소유자·URL·공개 범위·팀 제출 규칙·마감을 확인한다.

저장소를 초기화한 이후의 확인 명령은 다음과 같다. 아래 명령은 점검용이다. 공개용 사본에서 Git 작업을 진행하더라도 데이터·모델을 포함한 개인 작업 폴더 전체를 그대로 커밋하지 않는다.

```bash
git status --short
git check-ignore -v .env data/battery_modeling_ready.parquet results/final_svr/final_model.joblib
git ls-files -ci --exclude-standard
git diff --cached --stat
git diff --cached
```

`git ls-files -ci --exclude-standard`에 결과가 있으면 ignore 규칙에 해당하지만 이미 추적 중인 파일이다. `.gitignore`는 기존 추적 파일을 자동 제거하지 않는다. 파일을 확인한 후 필요한 정확한 경로에만 `git rm --cached <확인한 파일>`을 적용한다. 로컬 삭제나 Git 이력 전체 수정은 별도 판단이 필요하므로 무조건 전체 캐시를 지우는 명령은 사용하지 않는다. [GitHub ignore 안내](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files).

## 6. GitHub 제한과 보안 사고 대응

GitHub 일반 Git 저장소는 50MiB 초과 파일에 경고하고 100MiB 초과 파일을 차단한다. 브라우저 직접 업로드는 파일당 25MiB 제한이다. Git LFS를 쓰더라도 비밀정보나 재배포 권한 문제가 해결되는 것은 아니다. [GitHub 대용량 파일 안내](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

키가 이미 공개되었다면 먼저 폐기·회전하고, 파일 삭제나 `.gitignore` 추가만으로 과거 커밋에서 제거되었다고 판단하지 않는다. 기존 이력·포크·클론에 값이 남을 수 있다. 실제 이력 정리는 영향 범위를 확인하고 별도 승인 후 진행한다. [GitHub 민감정보 제거 안내](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
