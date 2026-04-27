# Phase 1-1 — 이미지 GUI 모니터링

- **상태**: ✅ 완료·검토대기 (2026-04-26)
- **시작일**: 2026-04-26
- **완료일**: 2026-04-26 (코드/테스트 green, vendored 모델로 E2E + GUI 스모크 통과)
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: `docs/decisions/20260426_gui-framework-pyqt6.md`, `docs/decisions/20260426_vendor-model-code.md` (당일 후반에 `consume-terravision-editable` 을 supersede)

## 목표

학습 체크포인트(`SCVTerraVision/outputs/checkpoints/dinov3_detr_base_full/epoch_050.pt`)를 PyQt6 GUI 에서 로드해, (a) 단일 이미지 또는 (b) 폴더(CODa `2d_rect/cam0/<SEQ>/`) 를 열고 detection 결과를 시각화한다.

## 확정된 결정

1. **모델 코드 vendoring (당일 후반 변경)** — 초기엔 SCVTerraVision 을 editable install 로 import 하려 했으나, `pyproject.toml` 의 `packages=[]` 문제 + sys.path 우회 helper + 브랜치 체크아웃 의존 등 마찰이 누적되어 ADR 변경. SCVTerraVision 의 inference 관련 코드 (`models/backbone/dinov3_backbone.py`, `models/detection/detr_head.py`, `train_detection.letterbox_resize` + IMAGENET 상수, `coda_taxonomy.yaml`) 를 SCVTerraScope 안에 vendor. 자세한 경위는 `docs/decisions/20260426_vendor-model-code.md`.
2. **HF DeformableDetrImageProcessor.post_process_object_detection** 을 그대로 사용 (eval_detection.py 와 동일 경로). threshold=0.0 으로 강제 호출 후 GUI 슬라이더에서 표시 필터만 적용.
3. **추론은 별도 QThread** (`gui/worker.py`). GUI 메인 루프는 시그널만 받음.
4. **클래스 라벨 = 패키지 내장 `src/scvterrascope/data/coda_taxonomy.yaml`** (vendor 사본).

## 체크리스트

- [x] `src/scvterrascope/model/{backbone,detr_head}.py` — SCVTerraVision DINOv3 + Deformable DETR architecture vendored.
- [x] `src/scvterrascope/data/coda_taxonomy.yaml` — 16-class operational taxonomy 사본 (operational_classes 만 유지).
- [x] `src/scvterrascope/labels.py` — bundled YAML 우선 로드 (importlib.resources). sibling/ENV 탐색 helper 제거.
- [x] `src/scvterrascope/inference/preprocess.py` — letterbox + IMAGENET 상수 inline. SCVTerraVision import 없음.
- [x] `src/scvterrascope/inference/engine.py` — vendored `scvterrascope.model.DinoV3DeformableDetr` 사용.
- [x] `src/scvterrascope/visualization/draw.py` — bbox + 클래스명 + score 라벨, HSV 16색 팔레트, threshold/class_filter, highlight.
- [x] `src/scvterrascope/gui/config.py` — YAML → AppConfig.
- [x] `src/scvterrascope/gui/worker.py` — `InferenceWorker(QThread)`.
- [x] `src/scvterrascope/gui/widgets/{image_canvas,control_panel,results_table,file_list,performance_panel}.py`
- [x] `src/scvterrascope/gui/main_window.py` — 메뉴/도크/시그널 와이어링/Export.
- [x] `src/scvterrascope/gui/__main__.py` + `scripts/launch_monitor.py` — 엔트리포인트.
- [x] `tests/{conftest.py, test_labels.py, test_preprocess.py, test_visualization.py, test_engine_imports.py}` — 비-Qt/비-torch 단위 테스트 (15 passed, 2 skipped).
- [x] `docs/runbooks/phase1-1_launch.md` — venv/HF_TOKEN/SCVTerraVision 브랜치 체크아웃/GUI 사용법.
- [ ] **(후속)** GPU 환경 실모델 실행 + 사용자 육안 검증 → 본 문서 검증 로그 append.

## 산출물

- `src/scvterrascope/` 전체 (labels, inference/, visualization/, gui/)
- `scripts/launch_monitor.py`
- `tests/` 5개 파일
- `docs/runbooks/phase1-1_launch.md`

## 검증 로그

### 단위 테스트 (vendor 후 최종, 2026-04-26)

- venv `.venv/` 부트스트랩 (`python3 -m venv --without-pip` + get-pip.py — 시스템 `python3-venv`/`pip` 부재 우회).
- `.venv/bin/python -m pytest -q` → **19 passed, 0 skipped** (Python 3.10.12, pytest 9.0.3). vendor 결정으로 SCVTerraVision 가용성에 의존하던 conditional skip 2개가 무조건 실행되는 정상 테스트로 전환됨.

### E2E 추론 스모크 (2026-04-26, RTX 4060 Ti)

- 환경: torch 2.11.0+cu130, transformers **4.56.2** (training-side 와 동일), timm 1.x, huggingface_hub.
- 체크포인트: `epoch_050.pt` (1098 MB).
- 합성 720×1280 RGB 입력 → letterbox 1024 → forward → post_process_object_detection.
- 결과: 모델 로드 4.0s, 첫 forward (warm) 441 ms, raw detections=100, top score 0.21 (`barrier`). 합성 이미지라 confidence 가 낮은 건 의도된 결과.
- DINOv3 백본 캐시는 SCVTerraVision 학습 시 이미 받아져 있어 이번 세션에서 추가 다운로드 없음.

### GUI 오프스크린 스모크 (`QT_QPA_PLATFORM=offscreen`)

- `MainWindow(cfg)` 빌드 → 16-class taxonomy 로드 → 파일 로드 → `_ensure_engine()` → `worker.submit()` → `finished` signal 수신까지 551 ms 사이클.
- 추론 후 `draw_detections` 로 오버레이 PNG 출력: `outputs/phase1-1_synthetic_overlay.png` (bbox + 클래스 색상 라벨 정상 렌더링 확인).
- 합성 이미지 입력은 `outputs/phase1-1_synthetic_input.png` 로 함께 저장.

### PerformancePanel (성능 모니터링) — 추가 산출 (2026-04-26)

사용자 요청으로 status bar 의 단일 `infer: <ms>` 만으로는 부족 → 우측 dock 에 `PerformancePanel` 위젯 신설. 표시 항목:

- **Last frame**: preprocess / forward / postprocess (ms) + total + FPS + detection 수
- **Rolling avg (last 30)**: 평균 total / 평균 FPS / min–max range / 누적 샘플 수
- **GPU memory**: allocated / peak (during this inference) / total + progress bar (% peak/total)
- **Model / device**: device 이름 (예: NVIDIA GeForce RTX 4060 Ti) + 파라미터 수 + letterbox 입력 크기

엔진 측 변경: `InferenceResult` dataclass 에 `preprocess_ms`, `postprocess_ms`, `gpu_mem_alloc_mb`, `gpu_mem_peak_mb`, `gpu_mem_total_mb` 필드 추가 + `total_ms`, `fps` property. `predict()` 가 단계별 `time.perf_counter()` 측정 + CUDA `synchronize` 후 `torch.cuda.max_memory_allocated()` 스냅샷.

CODa seq 0 의 5 프레임 처리 측정 (RTX 4060 Ti, 오프스크린):

| 항목 | 값 |
|---|---|
| Last frame total | 224.9 ms (pre 29 / fwd 195 / post 0.6 ms) |
| Last frame FPS | 4.45 |
| Rolling avg (n=5) | 285.2 ms (range 218–531 ms — 첫 프레임 warmup outlier 노출) |
| GPU peak | 530 MB / 7783 MB total = 7% |
| Model params | 95.93 M |

테스트 추가: `test_inference_result_has_perf_fields` — 새 필드 + `total_ms`/`fps` property 동작 확인. **20 passed (회귀 0)**.

### 발견된 환경 제약 (pyproject 갱신 + runbook 반영 완료)

1. **transformers 4.56.x 핀 필수** — 5.x 에서 HF DeformableDetr 의 backbone shim 경로가 변경되어 첫 forward 에서 `[256, 768]` weight ↔ `[1, 2048, 32, 32]` input 채널 mismatch 가 발생. pyproject 에 `transformers>=4.56,<5.0` 추가.
2. **timm 1.x 필요** — transformers 4.56 의 `DeformableDetrForObjectDetection.__init__` 가 default backbone (TimmBackbone, ResNet) 를 즉시 빌드하기 때문. 이후 vendored shim 이 덮어쓰지만 import 자체에 timm 필요. pyproject 에 `timm>=1.0` 추가.
3. **huggingface_hub 1.x 비호환** — transformers 4.56 이 `cached_download` 를 import 하므로 huggingface_hub 0.x 가 필요. pyproject 에 `huggingface_hub>=0.24,<1.0` 추가.
4. **HF_TOKEN 비-interactive shell 미반영** — 표준 ubuntu `~/.bashrc` 가 `*) return;;` 으로 조기 리턴하므로 systemd / `bash -c` 환경에서는 토큰이 보이지 않음. 사용자에게 `~/.profile` 사용 또는 GUI 실행 명령에 inline prefix 권장 (runbook §2 갱신).
5. **SCVTerraVision 의존 마찰** — `pyproject.toml` 의 `packages=[]` + 브랜치 별 코드 위치 차이로 인해 editable install 만으로는 import 가 안 잡힘. 임시 sys.path 우회 helper 도 시도했으나 깔끔하지 않아 **모델 코드 vendor 로 방향 전환** (당일 후반). ADR `20260426_vendor-model-code.md`.

### (미실행) 사용자 GUI 실 실행

DISPLAY=:1 X11 가용. 다음 명령으로 띄울 수 있음:

```bash
HF_TOKEN=hf_xxxxxxxx TERRAVISION_ROOT=~/development/SCVTerraVision \
    /home/soobin/development/SCVTerraScope/.venv/bin/python -m scvterrascope.gui \
    --checkpoint /home/soobin/development/SCVTerraVision/outputs/checkpoints/dinov3_detr_base_full/epoch_050.pt
```

CODa tiny / 임의 RGB 이미지 1장으로 GUI 동작 + bbox 시각 검증 → 본 섹션에 캡처 append.

### 사용자 실측 후 append 양식

```text
2026-MM-DD — GPU 환경 실 실행:
  - 환경: <GPU 모델>, torch <ver>, transformers <ver>
  - 체크포인트: epoch_050.pt
  - 이미지: data/coda_samples/2d_rect/cam0/0/2d_rect_cam0_0_420.jpg
  - 추론 시간: <ms>
  - bbox 시각 확인: ✅
  - 임계값 슬라이더 즉시 갱신: ✅
  - 폴더 모드 (시퀀스 0 전체): ✅
  - 캡처: outputs/phase1-1_demo.png
```

## 알려진 한계

- SCVTerraVision 의 `letterbox_resize` 시그니처가 변경되면 본 코드의 `preprocess.py` 가 깨진다. 변경 발생 시 본 단계 재검토 필요.
- HF 401 (gated DINOv3) 은 `HF_TOKEN` 누락 또는 라이선스 미수락 시 발생 — runbook 7절에 안내.
- `InferenceEngine` 은 단일 이미지/단일 배치만 지원. 배치 처리는 Phase 2 또는 후속에서.

## 사용자 검토 결과

(사용자 작성 영역)
