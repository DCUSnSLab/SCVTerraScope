# SCVTerraScope

GUI 기반 추론 모니터링 도구 for [SCVTerraVision](https://github.com/marsberry/SCVTerraVision) object detection 모델.

## 개요

SCVTerraVision 에서 학습된 DINOv3 + DeformableDETR 인퍼런스 모델을 실제 입력 데이터에 적용하여 결과를 시각적으로 검증하기 위한 도구이다. 학습 코드(SCVTerraVision)와 시각화/모니터링 도구(SCVTerraScope)의 관심사를 분리한다.

- **Phase 1** — 정지 이미지 / 폴더 일괄 GUI 모니터링 (PyQt6)
- **Phase 2** — ROS 2 토픽 영상 스트림 실시간 모니터링

진행 상황은 [`docs/progress/README.md`](docs/progress/README.md), 마스터 플랜은 [`docs/PLAN.md`](docs/PLAN.md) 참고.

## 설치 (개발 환경)

```bash
# 1) 가상 환경
python -m venv .venv
source .venv/bin/activate

# 2) PyTorch — CUDA 환경에 맞게 별도 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) SCVTerraVision (모델 코드) editable install
pip install -e ../SCVTerraVision

# 4) SCVTerraScope (본 저장소)
pip install -e .[dev]
```

## 검증 데이터 (CODa tiny split)

Phase 1-1 GUI 추론 검증을 위해 CODa devkit 의 공식 다운로더로 샘플을 가져온다. 자세한 절차는 [`docs/runbooks/data_setup.md`](docs/runbooks/data_setup.md).

```bash
python3 scripts/fetch_coda_samples.py --split tiny --dry-run   # 명령만 미리보기
python3 scripts/fetch_coda_samples.py --split tiny             # 실 다운로드
```

다운로드 결과는 `data/coda_samples/2d_rect/cam0/<SEQ>/*.jpg` 에 저장된다.

## 환경 변수

- `HF_TOKEN` — DINOv3 백본 다운로드용 (HuggingFace gated). SCVTerraVision README 참고.

## 실행 (Phase 1-1 완료 후)

```bash
python -m scvterrascope.gui --checkpoint /path/to/epoch_050.pt
# 또는
scvterrascope-monitor --checkpoint /path/to/epoch_050.pt
```

## 라이선스

M
MIT — `LICENSE` 참조.
