# Phase 1-0 — 검증 데이터 샘플 확보

- **상태**: ✅ 완료·검토대기 (2026-04-26)
- **시작일**: 2026-04-26
- **완료일**: 2026-04-26 (사용자 다운로드 실행 + 검증 로그 append 대기)
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: 없음 (방향성은 `docs/decisions/20260426_consume-terravision-editable.md` 와 정합)

## 목표

SCVTerraScope dev 머신에서 Phase 1-1 GUI 추론 검증에 사용할 CODa 샘플 데이터를 가져온다. 학습 서버의 `/home/marsberry/dataset/coda-devkit/data/CODa_full` 는 다른 사용자 계정에 있어 본 머신(`soobin@…`)에서 접근 불가하므로, 공식 CODa devkit (`github.com/ut-amrl/coda-devkit`) 의 `download_split.py` 로 별도 다운로드한다.

## 결정 사항 (2026-04-26)

1. **출처**: `https://github.com/ut-amrl/coda-devkit` 의 `scripts/download_split.py` (공식 도구). Texas Data Repository DOI `10.18738/T8/BBOQMV` 가 백엔드.
2. **기본 split**: `tiny` — coda-devkit 가 제공하는 4단계 (`tiny` < `small` < `medium` < `full`) 중 최소.
3. **저장 위치**: `<SCVTerraScope>/data/coda_samples/`. `.gitignore` 로 데이터 자체는 커밋하지 않음.
4. **다운로드 도구**: `scripts/fetch_coda_samples.py` — coda-devkit 클론 + `download_split.py` 호출만 자동화하는 얇은 래퍼 (재구현 X).
5. **Python 환경 분리**: coda-devkit 가 Python 3.8–3.9 권장이므로 SCVTerraScope `.venv` 와 충돌 시 `--python <conda env>` 옵션으로 분리.

## 체크리스트

- [x] `.gitignore` 에 `data/`, `data/coda_samples/.gitkeep`, `data/coda-devkit/.gitkeep` 예외 처리
- [x] `data/coda_samples/.gitkeep`, `data/coda-devkit/.gitkeep` 생성
- [x] `configs/default.yaml` 에 `samples_dir: data/coda_samples` 추가
- [x] `scripts/fetch_coda_samples.py` 작성
  - [x] `--split` / `--sequence` 상호 배타, `--dest`, `--coda-devkit-dir`, `--python`, `--dry-run` 옵션
  - [x] coda-devkit clone 자동화 (`.gitkeep` 만 있는 placeholder 디렉토리는 안전하게 정리)
  - [x] download_split.py 실행 (cwd = devkit 디렉토리)
- [x] `docs/runbooks/data_setup.md` 작성 — tiny vs sequence 가이드, 트러블슈팅
- [x] 본 progress 문서 작성 + 대시보드 갱신
- [ ] **(사용자 실행)** `python3 scripts/fetch_coda_samples.py --split tiny` 으로 실 다운로드 → `data/coda_samples/` 트리 확인 → 본 문서 "검증 로그" 에 append (다운로드 사이즈, 시퀀스 수, 샘플 1장 PIL 디코딩 결과)

## 산출물

- `scripts/fetch_coda_samples.py`
- `docs/runbooks/data_setup.md`
- `data/coda_samples/.gitkeep`, `data/coda-devkit/.gitkeep`
- `.gitignore` · `configs/default.yaml` 갱신
- 본 문서

## 검증 로그

- `python3 scripts/fetch_coda_samples.py --help` → argparse usage 정상 출력. ✅
- `python3 scripts/fetch_coda_samples.py --split tiny --dry-run` → `git clone https://github.com/ut-amrl/coda-devkit.git data/coda-devkit` + `download_split.py -d <abs>/data/coda_samples -t split -sp tiny` 명령 시퀀스 생성 확인. ✅
- `python3 scripts/fetch_coda_samples.py --sequence 5 --dry-run --coda-devkit-dir /tmp/nonexistent-coda-clone` → 비기본 경로에 대해서도 정상 명령 생성. ✅
- 옵션 검증: `--split tiny --sequence 0` 동시 지정 → argparse 에러 (mutually exclusive). ✅
- 옵션 검증: `--sequence 99` (범위 외) → argparse 에러. ✅
- **(부분 실행, 2026-04-26)** dev 머신에서 `python3 scripts/fetch_coda_samples.py --split tiny` 시도:
  - `git clone https://github.com/ut-amrl/coda-devkit.git data/coda-devkit/` 성공.
  - `download_split.py` 가 `from tqdm import tqdm` 단계에서 `ModuleNotFoundError`. 시스템 Python(3.10.12) 에 `tqdm` 미설치, `pip` 자체도 부재. ⛔
  - 우회: 우리는 `download_split.py` 의 URL (`https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_tiny_split.zip`) 만 추출해 `curl -L` 로 직접 다운로드. confirm prompt(`input("Download X GB?")`) 도 우회됨.
  - 향후 사용자: `--python <venv>/bin/python` 으로 tqdm/requests 설치된 venv 인터프리터 지정하거나, `curl` 직접 사용.

### 다운로드 완료 (2026-04-26 20:20, 본 세션 직접 실행)

- 출처: `https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/splits/CODa_tiny_split.zip`
- 파일 크기: **9,108,343,009 bytes (9.1 GB)** — Content-Length 와 정확히 일치
- 다운로드 시간: 약 10분, 평균 ~15 MB/s
- 압축 해제: `unzip -q CODa_tiny_split.zip` → 총 7050 entries, 약 9.8 GB extracted
- 디스크 사용 (extracted): 6.4 GB (2d_rect 이미지) + 3.2 GB (3d_comp 라이다) + 31 MB (3d_bbox GT JSON) + 32 MB (3d_semantic) + 169 MB (poses) + 1.2 MB (calibrations) + 5.5 MB (timestamps) + 164 KB (metadata)
- **이미지 포맷: PNG** (CODa DATA_REPORT 는 `.jpg` 라 했지만 실제 tiny split 은 `.png` — SCVTerraVision 의 `coda_validation_coco.json` 도 `.png` 로 표기되어 있어 일치)
- 이미지 해상도: 1224×1024 (cam0 첫 프레임 PIL 디코딩 확인)
- 시퀀스 22개 모두 포함 (0–21), cam0 frame 분포: 5 (seq21) ~ 305 (seq20), 총 ~1620 프레임. cam1 동수.
- 최상위 디렉토리 `CODa_tiny/` 가 zip 에 포함되어 있어 `samples_dir` 은 `data/coda_samples/CODa_tiny` 로 설정.

### 실모델 추론 검증 (2026-04-26)

같은 세션에서 추론 모니터로 실 CODa 프레임 1장 (`CODa_tiny/2d_rect/cam0/0/2d_rect_cam0_0_401.png`) 처리:
- engine.load 4.2s (오프라인, HF_TOKEN 불필요 — `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 강제 환경에서도 동작)
- forward 414.7 ms (RTX 4060 Ti)
- score >= 0.3 검출 8건. Top: `bollard 0.929`, `bollard 0.878`, `sign 0.873`, `bollard 0.868`, `barrier 0.862`. UT 캠퍼스 풍경에 부합하는 클래스 분포 확인.
- 오버레이: `outputs/phase1-0_real_coda_overlay.png` (사용자 육안 검증용 캡처).

### 디스크 정리

`data/coda_samples/CODa_tiny_split.zip` (9.1 GB) 은 압축 해제 후 redundant. 디스크 회수 필요시:
```bash
rm /home/soobin/development/SCVTerraScope/data/coda_samples/CODa_tiny_split.zip
```
재다운로드 가능하므로 본 단계에서는 자동 삭제하지 않고 사용자 판단에 위임.

### 사용자 실측 후 append 양식

```text
2026-MM-DD — split=tiny 다운로드:
  - 소요 시간: <분>
  - 디스크 사용: $(du -sh data/coda_samples)
  - 시퀀스 수: $(ls data/coda_samples/2d_rect/cam0 | wc -l)
  - 첫 샘플: $(ls data/coda_samples/2d_rect/cam0/*/*.jpg | head -1)
  - PIL 로드: size=(1224, 1024), mode=RGB ✅
```

## 사용자 검토 결과

(사용자 작성 영역)
