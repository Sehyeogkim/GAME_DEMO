# WEMINE Arcade 데이터 계약 — `wemine.arcade.v1`

8개 게임이 모두 같은 형식으로 기록한다. 게임은 장면·규칙·chunk 판정만 정의하고, 기록·분할·검증·로딩은 공통 코어(`arcade/core/`)가 한다.

## 가설과 이 형식의 관계

```text
episode = 학습 가능한 action chunk  +  게임 재미용 action chunk
          (split = "training")         (split = "other")
```

모든 프레임은 빠짐없이 정확히 하나의 chunk에 속한다. chunk는 지우지 않고 `split`으로 구분한다. 학습 loader는 `split == "training"`만 읽는 것이 기본값이다.

## 한 행(row)의 의미 — 20 Hz

```text
행 t:  관측 o_t (행동 전)  →  행동 a_t 실행  →  phase/chunk 결과
```

행동 입력에는 미래 관측이 섞이지 않는다. `timestamp`는 행동 전 시뮬레이션 시간이다.

## 에피소드 폴더

```text
runs/arcade/<game>/<episode_id>/
  metadata.json   게임·로봇·카메라·행동/상태 규격·seed·조작자·종료 사유
  episode.hdf5    프레임별 배열 (아래 표)
  chunks.json     chunk 목록과 split 판정 근거
  model.xml       그 에피소드의 MuJoCo 모델 전체 (재생용)
  manifest.json   파일별 크기·SHA256
```

저장 중에는 `.<episode_id>.partial/`에 쓰고 완료 시 원자적으로 rename한다. `.partial`이 남아 있으면 미완료 에피소드다.

## `episode.hdf5`

| 경로 | shape / dtype | 정책 입력? | 의미 |
|---|---|---|---|
| `obs/images/scene` | (T,) JPEG bytes | ✅ | 고정 3인칭 카메라 256×256, 게임 장식 숨김 |
| `obs/images/wrist` | (T,) JPEG bytes | ✅ | 손목 카메라 256×256, 게임 장식 숨김 |
| `obs/images/player` | (T,) JPEG bytes | ❌ | 플레이어가 본 게임 화면 (QA용) |
| `obs/joint_pos` | (T,7) f32 | ✅ | 팔 관절 위치 rad |
| `obs/joint_vel` | (T,7) f32 | ✅ | 팔 관절 속도 rad/s |
| `obs/gripper_width` | (T,1) f32 | ✅ | 두 손가락 간격 m (0–0.08) |
| `obs/eef_pos` | (T,3) f32 | ✅ | 그리퍼 중심(grasp_site) 월드 위치 m |
| `obs/eef_quat` | (T,4) f32 | ✅ | 그리퍼 자세 wxyz |
| `action/command` | (T,7) f32 | 정책 출력 ① | 정규화 delta 명령 [dx,dy,dz,droll,dpitch,dyaw,grip], 각 −1..1, grip +1=닫기 |
| `action/eef_target_pos` | (T,3) f32 | 정책 출력 ② | 명령 적용 후 절대 말단 목표 m |
| `action/eef_target_quat` | (T,4) f32 | 정책 출력 ② | 명령 적용 후 절대 말단 목표 자세 wxyz |
| `action/gripper` | (T,1) f32 | 정책 출력 ②③ | 그리퍼 목표 0=열림, 1=닫힘 |
| `action/joint_target` | (T,7) f32 | 정책 출력 ③ | IK가 낸 관절 위치 목표 rad |
| `action/source` | (T,) u8 | 필터 | 0=operator(사람/autopilot), 1=game_script(게임이 팔을 움직임) |
| `action/assist` | (T,) bool | 필터 | 게임이 물체에 비물리적 힘·순간이동을 가함 |
| `action/operator_active` | (T,) bool | 필터 | 조작자가 실제로 입력 중이었음 |
| `privileged/qpos` `qvel` | (T,nq) (T,nv) f64 | ❌ | 전체 상태, 재생·평가 전용 |
| `privileged/mj_state` | (T,ns) f64 | ❌ | MuJoCo integration state, 재생 전용 |
| `privileged/objects` | (T,N,7) f64 | ❌ | 게임 물체 pos+quat(wxyz), 이름은 metadata `objects` |
| `time/timestamp` | (T,) f64 | — | 행동 전 sim 시간, 정확히 0.05 s 간격 |
| `time/wall_time` | (T,) f64 | — | 실제 시각 (unix s) |
| `time/frame_index` | (T,) i64 | — | 0부터 연속 |
| `game/phase` | (T,) u8 | — | metadata `phases`의 인덱스 |
| `game/score` | (T,) f32 | — | 누적 점수 |
| `chunk/index` | (T,) i32 | — | 이 행이 속한 chunk 번호 |

정책 출력은 ①②③ 중 학습 방법에 맞게 하나를 고른다. 셋은 같은 실행 결과를 다른 좌표로 표현한 것이다.

정책 상태 벡터 `obs.state`(22)는 `joint_pos(7) + joint_vel(7) + gripper_width(1) + eef_pos(3) + eef_quat(4)` 순서로 loader가 이어 붙인다. `privileged/*`는 절대 정책 입력에 쓰지 않는다.

## `chunks.json`

```json
{"chunks": [
  {"index": 0, "start": 0, "end": 74,           // [start, end) 행 범위
   "skill": "grasp_lift", "instruction": "Pick up the toy and lift it.",
   "trainable": true,                           // 게임이 선언한 학습 대상 기술 여부
   "success": true, "evidence": {...},          // 물리 판정 근거 (접촉, 높이, 유지 시간 등)
   "split": "training", "other_reason": null,
   "operator_frames": 74, "script_frames": 0, "assist_frames": 0, "active_ratio": 0.93}
]}
```

### split 판정 — 코어가 하며 게임이 바꿀 수 없음

`training`은 아래를 **모두** 만족할 때만:
1. 게임이 `trainable=true`로 선언한 기술 구간
2. 물리 판정으로 `success=true`
3. 모든 행이 `source=operator`, `assist=false`
4. 길이 ≥ 5행

그 밖은 전부 `other`이며 `other_reason`은 다음 중 하나:

| other_reason | 뜻 |
|---|---|
| `game_fun` | 재미용 구간 (조준·발사·연출·점수) |
| `failed_attempt` | 학습 대상 기술이었지만 실패·중단 |
| `assisted` | 게임 스크립트나 보조가 끼어듦 |
| `too_short` | 5행 미만 |
| `setup` / `idle` | 게임이 선언한 준비·대기 구간 |

`other`도 삭제하지 않는다. 실패 데이터가 필요하면 loader 필터만 바꾼다.

## `metadata.json` 주요 필드

`schema_version`, `game`, `game_version`, `episode_id`, `seed`, `operator_kind`(`human`|`autopilot`), `input_device`, `control_hz`, `physics_dt`, `action_spec`, `state_spec`, `cameras`(해상도·fovy·intrinsics·고정 카메라 extrinsics·손목 카메라 부착 위치), `objects`, `phases`, `skills`, `end_reason`(`round_complete`|`reset`|`disconnect`|`max_frames`|`stop`), `num_frames`, `summary`(chunk별 개수).

`operator_kind=autopilot`은 게임별 스크립트 전문가가 **사람과 같은 명령 인터페이스**로 조작한 데이터다. 파이프라인과 학습 가능성 검증용이며, 사람 데이터의 품질을 대신 입증하지 않는다. loader는 기본으로 둘 다 읽되 `operator_kind` 필터를 제공한다.

## 무엇을 증명하고 무엇은 못 하나

- `verify`: 파일·shape·시간·chunk 범위·split 규칙·해시 일관성.
- `train_bc` + `eval_policy`: `training` chunk만으로 학습한 정책이 **새 seed**에서 그 기술을 혼자 성공하는 비율을, 학습 안 한 기준 정책과 비교.
- 못 하는 것: 사람 데이터의 품질, 실물 로봇 전이, 대규모 정책 성능.
