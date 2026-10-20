# Arcade 게임 작성 가이드 (에이전트용)

먼저 읽을 것: `arcade/docs/data-schema.md`, `arcade/core/game.py`, 기준 구현 `arcade/games/claw_prize/game.py`.

## 규칙

1. **본인 게임 폴더만 수정**: `arcade/games/<name>/` (`__init__.py`, `game.py`, `README.md`). `arcade/core/`, 다른 게임, `plate_game/`, 저장소의 다른 곳은 수정 금지. 코어 버그나 필요한 기능은 보고서에 적고 게임 쪽에서 우회한다.
2. 게임 클래스 이름과 포트는 `arcade/games/__init__.py` 레지스트리에 이미 등록되어 있다. 그 이름 그대로 구현한다.
3. Python: 저장소 루트에서 `.venv/bin/python` 사용. 서버 실행·포트 점유·git commit 금지.
4. 제품 UI 문자열(HUD, message, instruction)은 영어. README는 한국어 가능.

## 게임이 정하는 것

| 항목 | 방법 |
|---|---|
| 물리 물체 | `objects_xml()` — freejoint body, `object_bodies()`에 이름 등록 (privileged 로그용) |
| **정책이 봐야 하는 표시**(목표 마커, 정답 위치) | `objects_xml()`에 `contype="0" conaffinity="0"` 시각 전용 geom으로 넣는다 (group 0). **decor에 넣으면 정책 카메라에 안 보인다.** |
| 게임 장식 | `decor_xml()` — 플레이어 화면 전용, 자동으로 충돌 없음. 학습 판단에 필요한 정보는 넣지 말 것 |
| 학습 기술 | `skills = {"skill_id": "성공 조건 설명"}` — trainable 기술만 |
| 현재 행의 chunk | `self.label = Label(skill, instruction, trainable, other_reason)` |
| chunk 종료 | `self.finish_chunk(success, **evidence)` — 이번 행까지 포함해 닫힘. 다음 행부터 새 chunk |
| 라운드 종료 | `self.end_round("round_complete", message)` |

## chunk 규칙 (반드시 지킬 것)

- 학습 chunk(`trainable=True`)는 **사람이 직접 조작하는 구간**이다. 그 안에서 순간이동, 외력, `game_script` 금지. 코어가 감지해 자동으로 `other/assisted`로 내린다.
- 성공 판정은 **물리 근거**로: 양손가락 접촉(`self.finger_contacts`, `self.grasped`), 높이, 위치 오차, 기울기(`self.body_upright`), 정지 유지 시간 등. evidence에 수치를 남긴다.
- 실패도 닫는다: 예) 그리퍼를 다시 열었는데 성공 못 함 → `finish_chunk(False, reason=...)`. 실패 chunk는 `other/failed_attempt`로 보존된다.
- 재미 구간은 `Label(..., trainable=False, other_reason="game_fun")`. 여기서는 보조를 써도 된다:
  - 게임이 팔을 움직이면 `before_action`에서 `StepControl(command, source="game_script")`
  - 물체에 힘/속도를 주면 `substep()`에서 `self.data.xfrc_applied[...]`를 쓰고 `True` 반환 (또는 `StepControl(assist=True)`)
- label이 바뀌면 코어가 chunk를 자동으로 닫는다. trainable chunk를 `finish_chunk(True)` 없이 label 변경으로 닫으면 실패로 기록된다.
- **품질 연결**: 학습 기술을 잘할수록 재미 구간 결과가 좋아지게 설계한다 (가설의 핵심).

## 로봇과 작업 공간

- 테이블 윗면 z=0.74, 테이블 x 범위 −0.06..1.18, y −0.55..0.55. 로봇 베이스 (−0.15, 0, 0.74).
- 그리퍼 중심(grasp_site) 이동 범위 기본 x 0.15..0.85, y −0.42..0.42, z 0.752..1.25. 실제로 잘 닿는 곳은 x 0.25..0.70, |y| < 0.35.
- 그리퍼는 위에서 아래로 향함(roll/pitch 고정), yaw만 회전. yaw=0일 때 손가락은 world y축 방향으로 벌어진다. 최대 벌림 0.08 m → 잡을 면 폭은 0.06 m 이하로.
- 물체 질량 0.03–0.15 kg, `friction="1.8 0.02 0.001" condim="4" solref="0.004 1"`이 잘 잡힌다.
- 명령 1.0 = step당 1.2 cm, 20 Hz.

## autopilot (필수)

`autopilot()`은 privileged 상태를 보고 **사람과 같은 7D 명령**을 내는 스크립트 전문가다. 학습 가능성 검증 데이터를 만든다.
- `self.steer_to(point, yaw=...)` 사용, 순간이동 금지, 작은 노이즈 추가.
- 헤드리스 20 seed 중 **16개 이상 `round_complete`** 목표. 한 라운드 300–1200 프레임 권장.
- 막히면 재시도하는 로직을 넣어 무한 대기를 피한다.

## 검증 절차 (보고 전 필수)

```sh
cd GAME_DEMO   # repository root
# 1) 빠른 헤드리스 성공률 (렌더 끔)
.venv/bin/python - <<'EOF'
from arcade.games import load_game
from arcade.core.session import GameSession
g = load_game("<name>"); s = GameSession(g, record=False, render_player=False, render_policy=False)
ok = 0
for seed in range(20):
    s.reset(seed=seed); n = 0
    while not g.round_over and n < g.max_round_frames:
        c, b = g.autopilot(); s.step(c, b); n += 1
    ok += g.round_end_reason == "round_complete"
print(ok, "/20")
EOF
# 2) 기록 + 검증 (verified=true 여야 함)
.venv/bin/python -m arcade.tools.generate --game <name> --episodes 3
# 3) 화면 확인: 정책 카메라(scene, wrist)와 플레이어 카메라를 PNG로 저장해 직접 본다.
#    session.render_policy_views() / session.render_player_frame()
#    목표 마커와 물체가 scene 카메라에 보이는지, 플레이어 화면이 재밌어 보이는지 확인.
```

## 보고서 형식

1. 게임 규칙 요약, 학습 chunk / 재미 chunk 표, 성공 판정 수치
2. 품질 연결이 어떻게 작동하는지
3. autopilot 성공률 (n/20), 평균 프레임 수
4. `generate` 결과 (verified, chunk 수)
5. 사용한 보조(assist/game_script)와 위치
6. 알려진 문제, 코어에 요청할 것
