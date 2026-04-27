# Runbook — Phase 1-1 GUI 실행

본 runbook 은 학습된 체크포인트(`epoch_050.pt` 등)를 SCVTerraScope GUI 에서 열어 검증할 때 필요한 모든 사전 조건을 정리한다. SCVTerraScope 은 **모델 코드를 자체 vendor** 하므로 SCVTerraVision 저장소 자체는 더 이상 import 시점에 필요하지 않다 — 체크포인트 파일(.pt) 만 있으면 된다 (ADR `20260426_vendor-model-code.md`).

## 1) Python venv + 의존성

dev 머신(2026-04-26)에는 `python3-venv` apt 패키지와 `pip` 자체가 시스템 Python 에 없어 기본 `python3 -m venv` 가 깨진다. 우회:

```bash
cd ~/development/SCVTerraScope
python3 -m venv .venv --without-pip
curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py            # venv 안에 pip 부트스트랩
source .venv/bin/activate
python -m pip install --upgrade wheel

# PyTorch — 환경의 CUDA 버전에 맞게 설치. 기본 인덱스의 최신 wheel 이
# CUDA 12/13 자동 동봉되므로 보통 다음 한 줄로 충분.
pip install torch torchvision

# SCVTerraScope (본 저장소). transformers/timm/huggingface_hub 도
# pyproject 가 알아서 적절한 버전을 핀.
pip install -e .[dev]
```

> **transformers 버전 고정**: vendored DINOv3 backbone shim 은 HF DeformableDetr 의 4.56.x 내부 구조에 의존한다. transformers 5.x 가 깔리면 첫 forward 에서 `2048-vs-768 채널` 에러. SCVTerraScope pyproject 가 `transformers>=4.56,<5.0` 을 강제하므로 보통 자동 해결.

## 2) HF_TOKEN — **불필요**

학습된 체크포인트(`epoch_050.pt`) 가 backbone weights 까지 모두 포함하므로, SCVTerraScope 는 HuggingFace Hub 에 접속하지 않는다. `src/scvterrascope/model/backbone.py` 가 vendored config dict + `AutoModel.from_config()` 로 architecture 만 빌드한 뒤 checkpoint 로 weights 를 채운다.

→ `HF_TOKEN` 설정 불필요. `HF_HUB_OFFLINE=1` 환경에서도 정상 동작. 네트워크 끊긴 dev 머신/배포 머신에서도 GUI 가 그대로 뜬다.

(**참고**: SCVTerraVision 학습 시점에는 사전학습 weights 다운로드가 필요했고, 따라서 학습 환경의 `~/.cache/huggingface/` 에 327MB 가 남아있을 수 있다. SCVTerraScope 입장에서는 그 캐시도 사용하지 않는다.)

## 3) 검증 데이터 (CODa tiny)

`docs/runbooks/data_setup.md` 절차에 따라 `data/coda_samples/` 에 CODa tiny split 을 다운로드한다 (~9.1 GB). 단일 이미지만 빠르게 보고 싶으면 어떤 RGB 이미지(jpg/png)든 GUI 에 열면 된다 — 좌표 정합은 학습 시 1024 letterbox 와 동일하다.

## 4) 체크포인트 경로

기본값:
- `~/development/SCVTerraVision/outputs/checkpoints/dinov3_detr_base_full/epoch_050.pt` (Phase 1-2b 베이스라인 mAP=0.623)

`configs/default.yaml` 의 `checkpoint_path` 를 위 경로로 채우거나, GUI 의 Engine → Checkpoint → Browse 로 직접 선택한다. 체크포인트 파일 자체만 있으면 되고, SCVTerraVision 저장소 전체 체크아웃은 불필요.

## 5) GUI 실행

```bash
# 옵션 A — 패키지 엔트리포인트
scvterrascope-monitor --checkpoint ~/development/SCVTerraVision/outputs/checkpoints/dinov3_detr_base_full/epoch_050.pt

# 옵션 B — 모듈 형태
python -m scvterrascope.gui --checkpoint <path>

# 옵션 C — 미설치 상태에서 (개발용)
python scripts/launch_monitor.py --checkpoint <path>
```

옵션:

| Flag | 효과 |
|---|---|
| `--config <path>` | 다른 YAML 설정 사용 (기본 `configs/default.yaml`) |
| `--device {auto,cuda,cuda:0,cpu}` | 디바이스 강제 (기본 auto = CUDA 가능 시 CUDA) |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | 콘솔 로그 레벨 |

## 6) GUI 사용

1. **File → Open Image** 또는 **Open Folder** 로 입력 선택. 폴더는 재귀 탐색 (`2d_rect/cam0/<SEQ>/*.jpg` 같은 CODa 레이아웃 그대로 OK).
2. 좌측 Files 탭에서 이미지 클릭 → 메인 캔버스 갱신.
3. 좌측 Controls → Engine → Checkpoint Browse 또는 직접 입력.
4. **Run Inference** 클릭 → 별도 QThread 에서 모델 로드 후 추론. 첫 클릭은 모델 로드(수 초~수십 초) 포함.
5. **Score ≥ 슬라이더**: 추론 재실행 없이 표시 박스만 즉시 필터.
6. **Classes 체크박스**: 클래스별 ON/OFF 토글.
7. **Detections 테이블 행 클릭**: 캔버스에서 해당 박스 두꺼운 강조 표시.
8. **File → Export Result**: 현재 이미지 + 오버레이 PNG + JSON 예측을 폴더에 저장.

## 7) 트러블슈팅

| 증상 | 조치 |
|---|---|
| `RuntimeError: ... expected input ... 768 channels, but got 2048 channels` | transformers 5.x 가 깔려 있음. `pip install 'transformers<5.0'` 로 다운그레이드 (SCVTerraScope pyproject 가 자동 핀하지만 별도 환경에 5.x 가 선행 설치되면 충돌). |
| `TimmBackbone requires the timm library` | `pip install timm`. SCVTerraScope pyproject 가 의존성에 명시하므로 보통 자동. |
| `cached_download` ImportError 또는 huggingface_hub 1.x 호환 오류 | `pip install 'huggingface_hub<1.0'`. transformers 4.56 이 cached_download 를 import 하므로 1.x 와 호환되지 않음. |
| HF 401/네트워크 에러가 나타남 | SCVTerraScope 는 Hub 호출을 안 하므로 정상적으로는 발생할 수 없다. transformers 가 다른 모듈을 자동 탐색하다 발생했다면 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 명시적으로 export 후 재실행. |
| GUI 첫 클릭 시 멈춤 | 모델 로드 중. `--log-level DEBUG` 로 진행 확인. |
| `ModuleNotFoundError: PyQt6` | venv 재활성화 + `pip install -e .[dev]` 재실행. |
| bbox 가 이미지를 벗어남 | letterbox/scale 불일치. SCVTerraScope `inference/preprocess.py` 의 letterbox 와 학습 측 letterbox 가 어긋난 경우. ADR `20260426_vendor-model-code.md` 의 sync 정책 참고. |
| `state_dict` missing/unexpected keys 폭주 | vendored 모델 architecture 가 학습 체크포인트와 어긋남. SCVTerraVision 측 architecture 가 변경되었다면 ADR 의 sync 절차 수행 필요. |
