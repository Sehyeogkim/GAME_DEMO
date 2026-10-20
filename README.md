# GAME_DEMO — WEMINE Arcade

게임 플레이 로그를 로봇 학습 데이터로 쓸 수 있는지 검증하는 MuJoCo 게임 8종입니다.

```text
episode = { 학습 가능한 action chunk  |  게임 재미용 action chunk }
            split = "training"           split = "other"
```

모든 라운드를 자동으로 기록합니다. 각 chunk는 공통 코어가 `training` 또는 `other`로 판정합니다. 기준은 학습 기술인지, 물리 판정으로 성공했는지, 조작자가 직접 움직였는지, 보조가 없었는지입니다. 학습에는 `training`만 씁니다.

## 설치 (Python 3.11)

```sh
git clone https://github.com/Sehyeogkim/GAME_DEMO.git
cd GAME_DEMO
python3.11 -m venv .venv          # 또는: uv venv .venv --python 3.11
.venv/bin/pip install -r requirements.txt
# Windows: .venv\Scripts\python 를 .venv/bin/python 대신 사용
```

모든 명령은 저장소 루트에서 실행합니다.

## 실행

```sh
# 게임 하나 (브라우저 http://127.0.0.1:8802)
.venv/bin/python -m arcade.core.server --game claw_prize

# 전체 게임 + 허브 (http://127.0.0.1:8800)
.venv/bin/python -m arcade.tools.launch

# autopilot 데이터 생성 + 자동 검증 → runs/arcade/<game>/<episode_id>/
.venv/bin/python -m arcade.tools.generate --game claw_prize --episodes 5

# 학습 (training chunk만) → 새 seed에서 폐루프 평가 → baseline과 비교
.venv/bin/python -m arcade.tools.train_bc --game claw_prize --steps 4000
.venv/bin/python -m arcade.tools.eval_policy --game claw_prize --policy runs/arcade/_models/claw_prize/<run>/policy.pt
.venv/bin/python -m arcade.tools.eval_policy --game claw_prize --baseline zero
```

**조작:** W/S 앞뒤, A/D 좌우, R/F 또는 휠로 위아래, Q/E 회전, Space 또는 좌클릭으로 그리퍼, Shift 정밀 모드, Enter 또는 우클릭으로 게임 액션. 입력 처리는 `arcade/core/static/input.js` 한 파일에 있으니 게임패드는 이 파일만 바꾸면 됩니다.

## 구조

```text
arcade/core/    장면·Panda IK·세션·기록기·검증기·loader·서버·브라우저 UI (게임 공통)
arcade/games/   게임별 폴더 (장면·규칙·chunk 판정·autopilot만 정의)
arcade/tools/   autopilot 데이터 생성, 런처, BC 학습, 폐루프 평가
arcade/docs/    data-schema.md (데이터 계약), game-authoring.md (게임 작성 규칙)
assets/downloaded/panda/   Franka Panda 모델 (MuJoCo Menagerie, Apache-2.0, SOURCE.txt 참고)
```

## 게임 상태 (2026-10-20)

| 포트 | 게임 | 학습 chunk | 재미 chunk | 상태 |
|---|---|---|---|---|
| 8801 | Domino Run | 도미노를 마커에 세우기 | 연쇄로 쓰러뜨리기 | 초안, 미검증 |
| 8802 | Claw Prize | 장난감 집어 들기 | 배출구에 넣고 희귀도 뽑기 | ✅ 완료·검증 (autopilot 20/20, verify 통과) |
| 8803 | Color Sort Rush | 블록을 같은 색 통에 넣기 | 콤보, 웨이브 연출 | 초안, 미검증 |
| 8804 | Tower Stack | 블록 쌓기 | 바람 테스트 | 초안, 미검증 |
| 8805 | Bowling Setup | 핀을 지정 위치에 세우기 | 공 굴리기 | 초안, 미검증 |
| 8806 | Shape Sorter | 모양 블록을 구멍에 끼우기 | 잠금 해제 연출 | 초안, 미검증 |
| 8807 | Drawer Heist | 서랍 열기, 보석 집기 | 금고 배달, 보상 뽑기 | 초안, 미검증 |
| 8808 | Push Golf | 퍽을 티로 밀기 | 퍼팅 | 초안, 미검증 |

7개 초안은 에이전트가 병렬로 만들다가 중간에 멈춘 코드입니다. import는 되지만 autopilot 성공률, 녹화 검증, 화면 확인은 하지 않았습니다. 이어서 만들 때는 `arcade/docs/game-authoring.md`의 규칙과 검증 절차를 따르고, Claw Prize(`arcade/games/claw_prize/game.py`)를 기준 구현으로 보세요.

## 남은 일

1. **7개 게임 완성·검증:** 헤드리스 20 seed 성공률 16 이상, `generate` 결과 verify 통과, 카메라 화면 확인.
2. **학습 가능성 실험 (핵심, 아직 안 함):**
   - autopilot 데이터 약 80 에피소드 → `training` chunk만으로 BC 학습 → 새 seed(100000번대)에서 성공률 측정 → zero/random baseline과 비교
   - `--split all`로 재미 구간까지 섞어 학습한 정책과 비교하면 서브 가설을 직접 검증할 수 있습니다.
   - `train_bc.py`의 augment 버그를 고친 뒤 아직 한 번도 실행하지 않았습니다.
3. **실시간 성능:** 렌더링이 병목입니다(Mac 기준 step당 약 35 ms). 데스크톱 GPU에서는 더 빠를 것으로 예상합니다.
4. **사람 플레이 데이터 수집**과 autopilot 데이터와의 비교.

원본 개발 저장소: `Sehyeogkim/WEMINE` 브랜치 `claude/eight-games-1020`의 `game1016/arcade/`.
