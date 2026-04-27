# ADR: GUI 프레임워크로 PyQt6 채택

- **상태**: 확정
- **결정일**: 2026-04-26
- **관련 단계**: Phase 0 (골격), Phase 1 (이미지 GUI), Phase 2 (ROS 2 스트림)

## 배경

SCVTerraScope 는 SCVTerraVision 학습 모델의 추론 결과를 가시화하는 도구이다. Phase 1 은 정지 이미지/폴더 단위 모니터링이지만, 후속 Phase 2 에서는 ROS 2 토픽 영상 스트림을 실시간으로 표시해야 한다. GUI 프레임워크 선택이 두 단계의 연속성을 좌우한다.

## 검토한 대안

| 대안 | 장점 | 단점 |
|---|---|---|
| **PyQt6 / PySide6** | 데스크톱 네이티브 위젯, `QThread`+signals/slots 로 추론/IO 비동기 처리 자연스러움, `QImage`/`QGraphicsView` 로 영상 표시 성능 충분, ROS 2 와 별도 spin thread 통합 패턴 정립됨 | 의존성 무게 (~50MB), 라이선스 (PyQt6 GPL/상용 듀얼 — PySide6 LGPL 로 대체 가능) |
| Gradio | 빠른 프로토타이핑, 웹 공유 용이 | 영상 스트리밍/ROS 통합 시 백엔드 별도 구성 필요, Phase 2 에서 재작성 부담 |
| Tkinter | 표준 라이브러리, 의존성 0 | 실시간 영상 위젯/줌·팬 표현력 부족, 멀티스레드 모델 빈약 |

## 결정

**PyQt6 를 채택한다.** 이유:

1. **두 Phase 동일 스택** — Phase 1 의 `image_canvas`, `control_panel`, 추론 워커가 Phase 2 ROS 토픽 스트림에서 그대로 재사용된다.
2. **비동기 추론** — `QThread` / `QThreadPool` + `pyqtSignal` 로 GUI freeze 없이 무거운 PyTorch forward 를 백그라운드로 돌리고 결과만 메인스레드에 송신한다.
3. **영상 표시 성능** — `QGraphicsScene` + `QPixmap` 갱신은 1024×1024 이미지를 30+ FPS 로 렌더 가능 (검증은 Phase 1 종료 시).
4. **ROS 2 통합 선례** — `rclpy` 의 spin 을 별도 스레드로 두고 PyQt 메인 이벤트 루프와 신호로 연결하는 패턴이 잘 알려져 있다.

라이선스 측면에서 후속 상용 배포 검토 시 **PySide6 (LGPL) 로의 전환은 거의 import 만 바꾸는 수준**이므로 리스크 헤지가 가능하다 — 코드 작성 시 PyQt 고유 API (예: `pyqtSignal` 시그니처) 를 최소화해 호환성을 유지한다.

## 영향 / 후속 작업

- `pyproject.toml` 의 runtime dep 에 `PyQt6>=6.6` 명시.
- Phase 1 의 모든 GUI 코드는 `src/scvterrascope/gui/` 하위에만 둔다.
- Phase 2 에서 `rclpy` 통합 시 별도 ADR 작성 (스레드 모델 · QoS · cv_bridge 의존 등).
- 향후 PySide6 전환 결정 시 별도 ADR 로 마이그레이션 비용 평가.
