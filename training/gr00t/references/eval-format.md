# Observation and Response Format Reference

## Table of Contents

- [Observation Format](#observation-format)
- [Response Format](#response-format)
- [N1.6 vs N1.5 Key Differences](#n16-vs-n15-key-differences)
- [LeIsaac Client Abstraction](#leisaac-client-abstraction)
- [LeIsaac API Reference](#leisaac-api-reference)
- [Eval Parameters](#eval-parameters)

## Observation Format

The policy server (`run_gr00t_server.py`) expects nested-dict observations. The modality
keys come from the checkpoint's `processor_config.json` — for `new_embodiment`:

| Modality | Keys | Shape | Dtype |
|---|---|---|---|
| `video` | `front`, `wrist` | `(B, T, H, W, C)` | `uint8` |
| `state` | `single_arm`, `gripper` | `(B, T, D)` | `float32` |
| `language` | `annotation.human.action.task_description` | `list[list[str]]` | — |

- `B` = batch size (typically 1 for inference)
- `T` = temporal window (typically 1)
- `H, W, C` = height, width, channels (224, 224, 3)
- `D` = dimensionality (5 for single_arm, 1 for gripper)

Numpy arrays must be serialized using the `MsgSerializer` protocol (`np.save` to BytesIO,
wrapped in `{"__ndarray_class__": True, "as_npy": bytes}`). See `gr00t/policy/server_client.py`.

## Response Format

The server returns `list[action_dict, info_dict]` via msgpack.

- `action_dict["single_arm"]`: shape `(1, 16, 5)` float32
- `action_dict["gripper"]`: shape `(1, 16, 1)` float32

The 16 in the second dimension is the action horizon — the number of future action steps
predicted per inference call.

## N1.6 vs N1.5 Key Differences

The language instruction key changed between versions:
- **N1.6**: `annotation.human.action.task_description` (note the `.action.` segment)
- **N1.5**: `annotation.human.task_description`

Using the wrong key causes the server's strict validation to fail with `{"error": "..."}`,
which the client then raises as `KeyError: 0` (see troubleshooting).

## LeIsaac Client Abstraction

The leisaac `Gr00t16ServicePolicyClient` abstracts the raw observation format. It accepts
a simpler observation dict with:
- `front`, `wrist` — camera images
- `joint_pos` — 6D state
- `task_description` — string

It handles the conversion to the nested format internally. `policy_inference.py` uses this
client under the hood.

## LeIsaac API Reference

Available policy clients in `leisaac.policy.service_policy_clients`:

| Client | Protocol | Version |
|---|---|---|
| `Gr00t16ServicePolicyClient` | ZMQ | N1.6 (requires `main` branch commit) |
| `Gr00tServicePolicyClient` | ZMQ | N1.5 (in v0.3.0 tag) |
| `LeRobotServicePolicyClient` | gRPC | LeRobot |
| `OpenPIServicePolicyClient` | WebSocket | OpenPI |

## Eval Parameters

Key parameters for `policy_inference.py`:

| Parameter | Description | Default |
|---|---|---|
| `--task` | LeIsaac task name (e.g. `LeIsaac-SO101-PickOrange-v0`) | required |
| `--eval_rounds` | Number of episodes (0 = run indefinitely, press R to reset) | required |
| `--policy_type` | `gr00tn1.6` for N1.6, `gr00tn1.5` for N1.5 | required |
| `--policy_host` | Policy server hostname | `localhost` |
| `--policy_port` | Policy server port | `5555` |
| `--policy_timeout_ms` | Timeout per inference call | `5000` |
| `--policy_action_horizon` | Action steps per inference (16 for GR00T) | required |
| `--policy_language_instruction` | Natural language task description | required |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--enable_cameras` | Required for vision-based policies | flag |
