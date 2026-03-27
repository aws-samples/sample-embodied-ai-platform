# Direct Policy Server Test

Verify the policy server responds to inference requests before launching the full simulation.
This uses GR00T's own `PolicyClient` from the training container (not the leisaac client,
which requires a running IsaacSim process to import — see note below).

## Run the test

```bash
ECR_URI=$EcrImageUri   # From Phase 3 output capture

ssh dcv-isaac "docker run --rm --network=host \
  --entrypoint /bin/sh \
  $ECR_URI \
  -c 'cd /workspace/gr00t-repo && /workspace/gr00t-repo/.venv/bin/python -c \"
import sys, numpy as np
sys.path.insert(0, \\\"/workspace/gr00t-repo\\\")
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host=\\\"localhost\\\", port=5555)
obs = {
    \\\"video\\\": {
        \\\"front\\\": np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
        \\\"wrist\\\": np.random.randint(0, 255, (1, 1, 224, 224, 3), dtype=np.uint8),
    },
    \\\"state\\\": {
        \\\"single_arm\\\": np.zeros((1, 1, 5), dtype=np.float32),
        \\\"gripper\\\": np.zeros((1, 1, 1), dtype=np.float32),
    },
    \\\"language\\\": {
        \\\"annotation.human.action.task_description\\\": [[\\\"pick up the orange\\\"]],
    },
}
result = client.get_action(obs)
for k, v in result[0].items():
    print(f\\\"{k}: shape={v.shape}, dtype={v.dtype}\\\")
print(\\\"Policy server test PASSED\\\")
\"'"
```

## Expected output

```
single_arm: shape=(1, 16, 5), dtype=float32
gripper: shape=(1, 16, 1), dtype=float32
Policy server test PASSED
```

## Why not use the leisaac client for this test?

`leisaac.__init__` eagerly imports `leisaac.tasks` -> `isaaclab_tasks` -> `isaaclab` ->
`omni.physics.tensors`, which is a Kit runtime extension only available inside a running
IsaacSim process. Even `from leisaac.policy.service_policy_clients import ...` triggers
this chain because `service_policy_clients.py` imports `leisaac.utils.constant`. The
leisaac client works correctly inside `policy_inference.py` (which runs under IsaacSim),
but cannot be imported in a standalone Python script.
