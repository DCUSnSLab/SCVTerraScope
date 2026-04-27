# Phase 2-1 — ROS 2 bag 영상 재생/추론 모듈 통합

- **상태**: ✅ 완료·검토대기 (2026-04-27)
- **시작일**: 2026-04-27
- **완료일**: 2026-04-27 (코드 + offscreen smoke green; 사용자 X11 GUI 검증 대기)
- **담당 PR**: (작성 후 링크)
- **관련 ADR**: 없음 (rosbags lib 채택은 ADR 미필요 수준)

## 목표

Phase 1-1 의 PyQt6 GUI 에 ROS 2 bag 입력 모드를 추가. 단일 이미지 / 폴더 / bag 셋의 입력 소스를 동일 GUI 에서 탭으로 전환할 수 있고, bag 모드에서는 step / autoplay / seek 으로 시간축 자유 탐색이 가능해야 함.

## 확정된 결정 (2026-04-27)

| 항목 | 선택 | 비고 |
|---|---|---|
| Bag 라이브러리 | **`rosbags`** (pure-Python, pip) | ROS 환경 source 불필요. SCVTerraScope venv 자체로 동작. |
| GUI 통합 | 좌측 dock 의 **상단 QTabWidget** `[Image][Folder][ROS Bag]` | Canvas/Performance/Detections/Controls 는 모드 무관 공용. |
| Playback 제어 | Step + Autoplay (speed 가변) + Seek bar | drop-frame 으로 inference rate 보다 빠른 timer 보호. |
| Topic 선택 | Auto-detect + 수동 드롭다운 override | 드롭다운에 모든 image topic 제시, 첫 번째 자동 활성. |
| 좌표 변환 | Phase 1-1 의 letterbox 그대로 (cropped+pad_top 모두 처리) | bag 의 `bgra8` 인코딩 → BGR→RGB 변환만 추가. |

## 아키텍처

```
src/scvterrascope/
├─ rosbag/                              ← 신규
│   ├─ __init__.py
│   ├─ reader.py                       BagReader: rosbags 래퍼 + sqlite-direct seek
│   └─ types.py                        BagFrame(idx, ros_time_ns, image_pil, …)
├─ gui/
│   ├─ main_window.py                  ← Files dock 제거, Input QTabWidget 신설
│   ├─ playback.py                     ← 신규: PlaybackController 상태머신
│   ├─ worker.py                       ← submit_image(pil, tag) + drop-frame 추가
│   └─ widgets/
│       ├─ image_tab.py                ← 신규: 단일 이미지 입력
│       ├─ folder_tab.py               ← 신규: 폴더 입력 + thumbnail 리스트
│       └─ ros_bag_tab.py              ← 신규: bag/topic/transport
└─ pyproject.toml                      ← `rosbags>=0.10` 의존성 추가
```

## 핵심 기술 디테일

### BagReader 의 random-access 최적화

rosbags 의 forward iterator 는 message 마다 BLOB(이미지 5MB) 을 읽어 들이므로 4000 frame seek 에 9초가 걸렸음. 해결:

1. **timestamp-only 인덱싱 (sqlite 직접)** — `open()` 시 한 번 `SELECT timestamp FROM messages WHERE topic_id=?` 로 BLOB 빼고 timestamp 만 메모리에 캐시. 5222 row 가 6.7ms.
2. **frame_at(idx)**: 캐시된 timestamp[idx] → `r.messages(start=ts)` 로 rosbags 가 sqlite 의 timestamp 인덱스로 점프. `next()` 한 번에 target 도달.
3. **step_forward**: 캐시된 forward iterator 유지 (`_active_iter`). 매번 재시작 X. autoplay 의 hot path.

벤치 결과 (5222-frame 5MB-per-frame bag):

| 작업 | 이전 | 이후 |
|---|---|---|
| `open()` | ~30s (eager scan) | **93ms** |
| 첫 frame_at(idx=0) | 메모리 인덱스 + 1 next | ~17ms |
| frame_at(idx=4000) | **9000ms** | **73ms** (130× 개선) |
| step_forward (cached) | 20ms | 18ms |

### PlaybackController 의 drop-frame

`QTimer` 가 1/(base_fps × speed) 마다 timeout. 매 tick:
1. `is_busy()` 가 True 면 (worker 가 inference 중) → frame skip. 아무 것도 emit 안 함.
2. False 면 `step_forward` 로 다음 frame 가져와 `frame_ready` emit.

이 덕분에 inference (200-600ms) 가 timer interval (33ms @ 30fps) 보다 느려도 큐가 쌓이지 않고 timer 가 자연스럽게 다음 사용 가능 frame 만 emit.

### InferenceWorker 의 두 입력 모드

기존 `submit(paths)` (folder mode) 와 직교하는 `submit_image(pil, tag)` 추가:
- 단일 슬롯(`_pending`) 방식 — 새 submit 가 오면 미처리분 덮어씀 → most-recent-only.
- 현재 inference 중이면 끝나고 슬롯에 남은 마지막 프레임 처리.
- `is_busy()` 로 PlaybackController 가 drop-frame 판정.

### BGRA8 디코딩

ZED 카메라는 4-채널 BGRA. numpy reshape → channel reorder → PIL RGB:
```python
arr = buf.reshape(h, w, 4)
arr = arr[:, :, [2, 1, 0]]  # BGRA→RGB (drop alpha)
pil = Image.fromarray(arr.copy(), 'RGB')
```

## 체크리스트

- [x] `rosbags>=0.10` pyproject 추가 + venv 설치
- [x] `src/scvterrascope/rosbag/{reader,types}.py` — BagReader + BagFrame
- [x] `tests/test_rosbag_reader.py` — 6 tests 통과 (실 SCV bag 사용, sqlite seek 포함)
- [x] `gui/worker.py` 의 `submit_image` + `is_busy` 추가
- [x] `tests/test_worker.py` — 3 tests (drop-frame 동작 검증)
- [x] `gui/playback.py` — PlaybackController 상태머신
- [x] `tests/test_playback.py` — 5 tests (attach/step/seek/play/busy-drop)
- [x] `widgets/{image_tab,folder_tab,ros_bag_tab}.py` — 3 입력 탭 위젯
- [x] `widgets/control_panel.py` 의 input groupbox 제거 (탭이 입력 담당)
- [x] `main_window.py` 재구성: Files dock → Input QTabWidget, BagFrame 라우팅 추가
- [x] **45 pytest passed** (이전 40 + 5 신규)
- [x] Phase 2 offscreen smoke: bag 열기 + 5 frame 추론 → 100 dets / frame, 558ms→227ms
- [x] X11 GUI 사용자 검증 대기 — 1500×1171 윈도우 표시됨

## 검증 로그

### 단위 테스트 (2026-04-27)

```
tests/test_rosbag_reader.py::test_bag_reader_lists_image_topics PASSED
tests/test_rosbag_reader.py::test_bag_reader_first_frame_decodes_to_rgb PASSED
tests/test_rosbag_reader.py::test_bag_reader_frame_at_random_access PASSED
tests/test_rosbag_reader.py::test_bag_reader_duration_and_start PASSED
tests/test_rosbag_reader.py::test_bag_reader_rejects_missing_path PASSED
tests/test_rosbag_reader.py::test_bag_reader_iter_skips_to_start_index PASSED
tests/test_playback.py — 5 passed (attach/step/seek/play/busy-drop)
tests/test_worker.py — 3 passed (submit_image / drop / submit paths still works)
전체: 45 passed, 7 warnings in 1.94s
```

### 통합 offscreen smoke (`QT_QPA_PLATFORM=offscreen`)

`260113_SCV_D2_Detection_01` bag (5222 frames, 175.2s) 에서:

```
frame_count=5222, initial canvas=(1920, 1080)
inferences: 5
  bag:.../image_rect_color:000000  total=558ms  dets=100  pad_top=0
  bag:.../image_rect_color:000001  total=239ms  dets=100  pad_top=0
  bag:.../image_rect_color:000002  total=231ms  dets=100  pad_top=0
  bag:.../image_rect_color:000003  total=227ms  dets=100  pad_top=0
  bag:.../image_rect_color:000004  total=230ms  dets=100  pad_top=0
PHASE 2 SMOKE OK
```

첫 프레임 558ms = warmup 포함, 이후 230ms 안정 (Phase 1-1 SCV 결과와 동일 수준).

### 사용자 X11 GUI 검증 양식 (수동)

```text
2026-MM-DD — 사용자 직접 실행:
  - Image 탭: __ (single image 추론 동작)
  - Folder 탭: __ (폴더 모드 동작)
  - ROS Bag 탭:
    - Open Bag → ~/data/260113_SCV_D2_Detection_01 → topic 자동 표시: ___
    - 첫 frame 표시 + bbox: ___
    - Step ⏭ +1 동작: ___
    - Autoplay ⏯ 동작 + drop-frame 부드러움: ___
    - Speed 토글 (0.5x/2x): ___
    - Seek bar 드래그: ___
    - PerformancePanel rolling avg 갱신: ___
  - 탭 전환 후 상태 보존: ___
```

## 후속 fix — autoplay 렌더링 병목 해결 (2026-04-27 후반)

### 사용자 보고

자동재생 시 frame interval 이 ~2초 (inference 200ms 의 10×). 사용자가 GUI 멀티스레드/렌더링 병목 의심.

### 원인 진단 (`scripts/bench_render.py`)

| 단계 | 1920×1080 | 1224×1024 |
|---|---|---|
| `pil_to_pixmap` PNG round-trip (현행) | **212 ms** / 호출 | 128 ms / 호출 |
| `pil_to_pixmap` numpy/QImage (제안) | **1.1 ms** | 1.2 ms |
| `draw_detections` (100 boxes) | 22 ms | 16 ms |
| 현재 cycle (raw set + draw + overlay set) | **442 ms** | 266 ms |
| Step B+C 적용 후 cycle | **24 ms** ✓ | 19 ms |

PNG encode/decode 가 호출당 200ms+ 이고, bag mode 가 frame 당 2번 호출 → 442ms / frame overhead 가 2초 latency 의 주범.

### 적용 fix

| Step | 위치 | 변경 |
|---|---|---|
| **B** | `widgets/image_canvas.py::pil_to_pixmap` | PNG round-trip → `numpy.asarray(pil) → QImage(Format_RGB888) → QPixmap.fromImage`. 190× 가속. |
| **C** | `main_window.py::_on_bag_frame` | raw 이미지 즉시 set_image 제거. 추론 완료 후 `_redraw_overlays` 가 한 번에 갱신. status 에 "(inferring…)" 피드백. |
| **D** | `visualization/draw.py` | RGB 입력 short-circuit (`convert("RGB").copy()` → `image.copy()`); module-level `_FONT_CACHE` 로 truetype 디스크 lookup 1회만. |

### End-to-end 측정 (5222-frame SCV bag autoplay, RTX 4060 Ti)

```
12 frames collected in 3.4s
  frame#0   engine=536.8  wall=   0.0  delta= —    (warmup)
  frame#1   engine=236.4  wall= 269.7  delta=269.7
  frame#2   engine=231.3  wall= 539.2  delta=269.5
  frame#3   engine=228.3  wall= 792.6  delta=253.4
  ...
  frame#11  engine=228.3  wall=2840.3  delta=252.2

steady-state delta:  median=254 ms  mean=256 ms  min=252  max=261
effective FPS:       3.91
```

frame interval 의 **inference (226-232ms) + GUI cycle (~25ms)** 합이 측정값과 정합. inference 200ms 한계까지 도달.

| 지표 | Before | After |
|---|---|---|
| Frame interval | ~2000 ms | **254 ms** |
| Effective FPS | < 0.5 | **3.91** |
| GUI thread overhead / frame | 442 ms | 24 ms (18× 개선) |

### 회귀

- `pytest -q`: 45 passed (변화 없음).
- 시각: bag frame 0 의 bbox 위치/색상은 fix 전후 동일 (numpy 경로 픽셀 정합 확인).
- Image / Folder 탭 동작 변화 없음 (raw set_image 유지 — 단발 응답성 우선).

## 알려진 한계 / 후속

- **rosbags forward iterator 의 BLOB 읽기 비용** — frame_at 의 캐시 hit 경로는 빠르지만 random seek 후 +1, +2 step 은 잠시 느릴 수 있음 (rosbags 가 BLOB 을 가져옴). 사용자가 거슬려 하면 sqlite-cursor 직접 운영으로 추가 최적화.
- **CompressedImage** 디코딩은 코드는 있으나 실 데이터로 미검증 (bag 에 raw Image 만 있음).
- **추론 결과를 새 bag 으로 녹화**: Phase 2-V2 또는 Phase 3 별도.
- **Lidar overlay (3D bbox)**: SCVTerraVision Phase 1-4 BEV 프로젝션과 묶어서 별도 단계.
- **Live ROS topic 구독** (실시간 카메라): Phase 3 멀티카메라/온보드와 함께.

## 사용자 검토 결과

(사용자 작성 영역)
