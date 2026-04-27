# Runbook — CODa 검증 샘플 다운로드

본 runbook 은 SCVTerraScope GUI 검증에 사용할 CODa 샘플 데이터를 dev 머신에 가져오는 절차를 설명한다. SCVTerraVision 학습 시 사용한 `/home/marsberry/dataset/coda-devkit/data/CODa_full` 는 다른 사용자 계정이라 본 머신에서 접근 불가하므로 별도 다운로드가 필요하다.

## 사전 준비

- Git, Python 3.10 (이미 셋업되어 있음).
- 디스크 여유 공간 — 아래 split 별 안내 참고.
- **`tqdm`, `requests` 가 호출될 Python 인터프리터에 설치되어 있어야 함** — coda-devkit 의 `download_split.py` 가 import 함. 시스템 Python 에 pip/tqdm 이 없으면 별도 venv/conda env 를 만든다.

```bash
# 옵션 A — venv (가장 가볍게)
python3 -m venv .venv-coda
source .venv-coda/bin/activate
pip install tqdm requests
deactivate
# 이후 fetch 스크립트는 --python .venv-coda/bin/python 으로 호출.

# 옵션 B — coda-devkit 전용 conda env (devkit 권장 환경)
conda create -n coda python=3.9 -y
conda activate coda
pip install tqdm requests
# 이후 fetch 스크립트는 --python ~/miniconda3/envs/coda/bin/python 으로 호출.
```

dev 머신(2026-04-26 시점)에서는 시스템 Python 에 `pip` 자체가 없어 시스템 import 만으로는 다운로드 불가했음 (`docs/progress/phase1-0_data_samples.md` 검증 로그 참고). 위 옵션 A 가 가장 빠른 해결책.

## 1) tiny split 다운로드 (기본)

가장 작은 split. 실 사이즈는 coda-devkit 문서에 미공개이므로, 다운로드 후 `du -sh data/coda_samples` 로 측정해 `docs/progress/phase1-0_data_samples.md` 검증 로그에 기록한다.

```bash
cd /home/soobin/development/SCVTerraScope

# 미리 한 번 dry-run 으로 호출 명령 확인
python3 scripts/fetch_coda_samples.py --split tiny --dry-run

# 실제 다운로드
python3 scripts/fetch_coda_samples.py --split tiny
```

스크립트가 자동으로 수행하는 일:

1. `data/coda-devkit/` 에 `git clone https://github.com/ut-amrl/coda-devkit.git` (기존 클론 있으면 재사용).
2. coda-devkit 의 `scripts/download_split.py -d <SCVTerraScope>/data/coda_samples -t split -sp tiny` 실행.

다운로드가 끝나면 `data/coda_samples/` 하위에 다음 레이아웃이 만들어진다:

```
data/coda_samples/
├─ 2d_rect/cam0/<SEQUENCE>/2d_rect_cam0_<SEQ>_<FRAME>.jpg     # 1224×1024 RGB JPG
├─ 2d_rect/cam1/<SEQUENCE>/2d_rect_cam1_<SEQ>_<FRAME>.jpg
├─ 3d_bbox/os1/3d_bbox_os1_<SEQ>_<FRAME>.json                  # GT 비교용 (Phase 1-1 필수 아님)
├─ calibrations/<SEQUENCE>/                                    # intrinsic/extrinsic
└─ metadata/<SEQUENCE>.json
```

## 2) 단일 시퀀스만 받기 (대안)

전 시퀀스(0–21) 중 하나만 받아 빠르게 시작하고 싶다면:

```bash
# 시퀀스 0 — ~17GB
python3 scripts/fetch_coda_samples.py --sequence 0
```

## 3) 다른 split (small / medium / full)

```bash
python3 scripts/fetch_coda_samples.py --split small      # tiny 다음 크기
python3 scripts/fetch_coda_samples.py --split medium
python3 scripts/fetch_coda_samples.py --split full       # ~1.5TB — 학습 서버 외 비권장
```

## 4) 다른 위치에 저장

```bash
python3 scripts/fetch_coda_samples.py --split tiny --dest /mnt/external/coda_samples
# 다운로드 후 configs/default.yaml 의 samples_dir 도 같은 경로로 수정.
```

## 5) 기존 coda-devkit 클론 재사용

이미 다른 위치에 coda-devkit 가 클론되어 있다면:

```bash
python3 scripts/fetch_coda_samples.py --split tiny \
    --coda-devkit-dir ~/repos/coda-devkit
```

## 6) Python 버전 충돌 시

coda-devkit 는 Python 3.8–3.9 환경에서 검증되었다. SCVTerraScope 의 Python 3.10 에서 download_split.py 가 실패하면 다음과 같이 conda env 의 Python 을 명시한다:

```bash
python3 scripts/fetch_coda_samples.py --split tiny \
    --python ~/miniconda3/envs/coda/bin/python
```

## 7) 다운로드 후 SCVTerraScope 설정 확인

`configs/default.yaml` 의 `samples_dir` 이 다운로드 위치와 일치하는지 확인:

```yaml
samples_dir: data/coda_samples
```

이미지 1장 디코딩 sanity check:

```bash
python3 -c "
from PIL import Image
import glob, os
samples = sorted(glob.glob('data/coda_samples/2d_rect/cam0/*/*.jpg'))
print(f'found {len(samples)} cam0 frames')
if samples:
    img = Image.open(samples[0])
    print('first sample:', samples[0], 'size:', img.size, 'mode:', img.mode)
"
```

기대 결과: `size: (1224, 1024)`, `mode: RGB`.

## 8) 의존성 누락 시

다음 에러가 나면 **인터프리터에 `tqdm` 또는 `requests` 가 없는 것**이다:

```
ModuleNotFoundError: No module named 'tqdm'
```

해결책:

```bash
# 새 venv 만들고 의존성 설치
python3 -m venv .venv-coda
source .venv-coda/bin/activate
pip install tqdm requests
deactivate

# 그 venv 의 python 으로 다시 실행
python3 scripts/fetch_coda_samples.py --split tiny --python .venv-coda/bin/python
```

## 트러블슈팅

| 증상 | 원인 / 조치 |
|---|---|
| `ModuleNotFoundError: No module named 'tqdm'` | 위 "8) 의존성 누락 시" 참고. |
| `pip: command not found` | 시스템 Python 에 pip 미설치. `python3 -m venv .venv-coda` 로 venv 생성 후 그 안의 pip 사용. apt 가능하면 `sudo apt install python3-pip` 도 가능. |
| `git clone` 실패 (네트워크) | 프록시·VPN 확인. 수동으로 `git clone https://github.com/ut-amrl/coda-devkit.git data/coda-devkit` 후 스크립트 재실행. |
| `download_split.py` 가 conda env 누락 에러 | `--python` 으로 conda env 인터프리터 지정 (`6) Python 버전 충돌 시` 참고). |
| Texas Data Repository (TDR) rate-limit / 느린 속도 | 시간대 변경 후 재시도. 부분 다운로드 시 download_split.py 가 resume 지원하는지 coda-devkit 측 README 확인. |
| 디스크 공간 부족 | `--dest` 로 큰 디스크 마운트로 변경. tiny → small → medium 순서로 점진 확장. |

## 자동화 한계 / 후속

- coda-devkit 의 정확한 tiny split 사이즈는 본 runbook 작성 시점(2026-04-26)에 미공개이므로 사용자 1회 실측 후 `docs/progress/phase1-0_data_samples.md` 검증 로그에 기록한다.
- TDR 직접 다운로드(개별 파일) 가 필요한 경우(예: 특정 프레임만) DOI `10.18738/T8/BBOQMV` 페이지를 수동 사용. 자동화는 후속 ADR 로 검토.
