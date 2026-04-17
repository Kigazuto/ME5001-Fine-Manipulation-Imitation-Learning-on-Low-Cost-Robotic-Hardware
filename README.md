# ME5001 Complete Project Tutorial: Koch Robotic Arm + ACT Imitation Learning

**Project Goal**: Implement imitation learning for fine manipulation on a low-cost Koch robotic arm using ACT (Action Chunking Transformer).

**Target Tasks**: High-precision insertion tasks such as battery insertion / USB insertion.

**This tutorial is based on the local repository `/home/kirito/ME5001` (LeRobot source code). All commands can be directly copied and executed.**

---

## Table of Contents

- [Phase A: Without Hardware](#phase-a-without-hardware)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Repository Installation](#2-repository-installation)
  - [3. Verify Installation](#3-verify-installation)
  - [4. Understanding the ACT Training Pipeline](#4-understanding-the-act-training-pipeline)
  - [5. Train ACT with Public Dataset (Quick Verification)](#5-train-act-with-public-dataset-quick-verification)
  - [6. Train ACT with Public Dataset (Full Training)](#6-train-act-with-public-dataset-full-training)
  - [7. Inspect Checkpoint](#7-inspect-checkpoint)
  - [8. Simulation Environment Evaluation](#8-simulation-environment-evaluation)
  - [9. Ablation Experiments (Optional)](#9-ablation-experiments-optional)
- [Phase B: With Hardware](#phase-b-with-hardware)
  - [10. Koch Hardware Overview](#10-koch-hardware-overview)
  - [11. Dependency Installation (Dynamixel)](#11-dependency-installation-dynamixel)
  - [12. Find Motor USB Ports](#12-find-motor-usb-ports)
  - [13. Configure Motor IDs](#13-configure-motor-ids)
  - [14. Update KochRobotConfig Ports](#14-update-kochrobotconfig-ports)
  - [15. Calibrate the Robotic Arms](#15-calibrate-the-robotic-arms)
  - [16. Teleoperation Test](#16-teleoperation-test)
  - [17. Configure Cameras](#17-configure-cameras)
  - [18. Record Dataset](#18-record-dataset)
  - [19. Inspect Recorded Data](#19-inspect-recorded-data)
  - [20. Replay Verification](#20-replay-verification)
  - [21. Train ACT with Your Own Data](#21-train-act-with-your-own-data)
  - [22. Real Robot Deployment and Evaluation](#22-real-robot-deployment-and-evaluation)
- [Phase C: Migrate to ME5001 Insertion Task](#phase-c-migrate-to-me5001-insertion-task)
- [Appendix: Common Errors and Fixes](#appendix-common-errors-and-fixes)

---

# Phase A: Without Hardware

> Goal: Complete the full pipeline of ACT training -> checkpoint saving -> simulation evaluation. No Koch hardware required.

## 1. Environment Setup

### What to Do
Ensure the system has the basic development environment ready.

### Commands to Run
```bash
# Confirm Python version >= 3.10
python3 --version

# Confirm pip is available
pip --version

# Confirm git is available
git --version

# If you have an NVIDIA GPU, confirm CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Success Criteria
- Python >= 3.10
- If you have a GPU, `torch.cuda.is_available()` returns `True`

### Troubleshooting
- Python version too low: Install Python 3.10+
- CUDA not available: Check that the NVIDIA driver and PyTorch CUDA version match

---

## 2. Repository Installation

### What to Do
Install the local LeRobot repository in editable mode, along with the ALOHA simulation environment (used for insertion task evaluation during the no-hardware phase).

### Why
- `pip install -e .` installs the LeRobot core (including ACT, datasets, training scripts)
- The `[aloha]` extra installs `gym-aloha>=0.1.1`, which provides the `AlohaInsertion-v0` simulation environment

### Commands to Run
```bash
cd /home/kirito/ME5001
pip install -e ".[aloha]"
```

### Additional Steps for Linux
If you encounter ffmpeg/opencv issues on Linux:
```bash
conda install -c conda-forge ffmpeg
pip uninstall opencv-python
conda install -c conda-forge "opencv>=4.10.0"
```

### Success Criteria
```bash
python -c "import lerobot; print(lerobot.__version__)"
# Should output a version number, e.g., 0.1.0

python -c "import gym_aloha; print('gym_aloha OK')"
# Should output gym_aloha OK
```

### Troubleshooting
- If `import lerobot` fails: Check that you ran `pip install -e .` in the `/home/kirito/ME5001` directory
- If `import gym_aloha` fails: Check that you used `".[aloha]"` instead of `"."`

---

## 3. Verify Installation

### What to Do
Confirm that all key modules of the ACT training pipeline can be imported correctly.

### Commands to Run
```bash
# Verify training entry point
python -c "from lerobot.scripts.train import train; print('train.py OK')"

# Verify evaluation entry point
python -c "from lerobot.scripts.eval import eval_main; print('eval.py OK')"

# Verify ACT policy
python -c "from lerobot.common.policies.act.modeling_act import ACTPolicy; print('ACTPolicy OK')"

# Verify ACT configuration
python -c "from lerobot.common.policies.act.configuration_act import ACTConfig; print('ACTConfig OK')"

# Verify dataset
python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; print('LeRobotDataset OK')"

# Verify simulation environment
python -c "from lerobot.common.envs.configs import AlohaEnv; print(f'AlohaEnv task={AlohaEnv().task}')"
# Should output: AlohaEnv task=AlohaInsertion-v0
```

### Success Criteria
All outputs show OK, and the last line displays `AlohaInsertion-v0`.

---

## 4. Understanding the ACT Training Pipeline

### What to Do
Understand the meaning of each parameter in the training command. No training is executed in this step.

### Key Files

| File | Purpose |
|------|---------|
| `lerobot/scripts/train.py` | Training entry point, main function `train(cfg: TrainPipelineConfig)` |
| `lerobot/configs/train.py` | `TrainPipelineConfig` definition, contains all training parameters |
| `lerobot/common/policies/act/configuration_act.py` | `ACTConfig`, all ACT hyperparameters |
| `lerobot/common/policies/act/modeling_act.py` | `ACTPolicy`, ACT model implementation |
| `lerobot/common/policies/factory.py` | `make_policy()`, policy instantiation |
| `lerobot/common/datasets/factory.py` | `make_dataset()`, dataset loading |
| `lerobot/common/envs/configs.py` | `AlohaEnv`, simulation environment configuration (action shape=(14,), fps=50) |

### Training Command Structure
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=<HF dataset ID> \       # Required: data source
  --policy.type=act \                        # Required: policy type
  --env.type=aloha \                         # Optional: simulation evaluation environment
  --output_dir=<output directory> \          # Optional: checkpoint save location
  --batch_size=8 \                           # Default 8
  --steps=100000 \                           # Default 100000
  --eval_freq=20000 \                        # Default 20000, set 0 to skip eval
  --save_freq=20000 \                        # Default 20000
  --policy.device=cuda                       # cuda / mps / cpu
```

### ACT Default Hyperparameters (from `ACTConfig`)
- `chunk_size=100`: Predict 100 action steps at once
- `n_action_steps=100`: Execute all 100 predicted steps
- `vision_backbone="resnet18"`: Visual encoder
- `dim_model=512`: Transformer hidden dimension
- `n_heads=8`: Number of attention heads
- `n_encoder_layers=4`: Number of encoder layers
- `n_decoder_layers=1`: Number of decoder layers
- `use_vae=True`: Use VAE
- `latent_dim=32`: VAE latent variable dimension
- `kl_weight=10.0`: KL divergence loss weight
- `optimizer_lr=1e-5`: Learning rate

---

## 5. Train ACT with Public Dataset (Quick Verification)

### What to Do
Run 100 training steps to verify the entire pipeline works. No GPU required, no need to download large amounts of data.

### Why
First confirm the code runs without errors, then proceed with long training.

### Commands to Run
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.type=act \
  --output_dir=outputs/train/act_test_100steps \
  --job_name=act_test \
  --batch_size=4 \
  --steps=100 \
  --eval_freq=0 \
  --save_freq=100 \
  --policy.device=cpu
```

**Parameter Explanation**:
- `--dataset.repo_id=lerobot/aloha_sim_insertion_human`: A public ALOHA insertion dataset on HuggingFace (collected via human teleoperation). **First run requires internet access for download**, then cached locally
- `--eval_freq=0`: Skip simulation evaluation (speeds up verification)
- `--steps=100`: Only run 100 steps
- `--policy.device=cpu`: Use CPU (works without a GPU)

### Success Criteria
```bash
ls outputs/train/act_test_100steps/checkpoints/last/pretrained_model/
# Should see:
# config.json
# model.safetensors
# train_config.json
```

### Troubleshooting
- `ModuleNotFoundError`: Go back to Step 2 to check the installation
- Network error: Confirm you can access huggingface.co, or set the `HF_ENDPOINT` environment variable
- `FileExistsError: Output directory already exists`: Delete `outputs/train/act_test_100steps/` or use a different `--output_dir`

---

## 6. Train ACT with Public Dataset (Full Training)

### What to Do
Perform full training on GPU with simulation evaluation enabled.

### Why
This is your first experiment for the ME5001 project: train ACT on the ALOHA insertion simulation and measure the success rate.

### Commands to Run
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.type=act \
  --env.type=aloha \
  --output_dir=outputs/train/act_me5001_insertion \
  --job_name=act_me5001_insertion \
  --batch_size=8 \
  --steps=50000 \
  --eval_freq=10000 \
  --save_freq=10000 \
  --policy.device=cuda
```

**Parameter Explanation**:
- `--env.type=aloha`: Evaluate the policy in the `AlohaInsertion-v0` simulation environment every 10000 steps
- `--steps=50000`: 50000 training steps (takes several hours depending on GPU)
- If you don't have a GPU, change to `--policy.device=cpu` and reduce `--steps`

### Training Logs
You will see output similar to:
```
INFO ... step: 200   loss: 0.1234   grad_norm: 1.56   lr: 1e-05   update_s: 0.45
INFO ... step: 400   loss: 0.0987   grad_norm: 1.23   lr: 1e-05   update_s: 0.44
```

Every `eval_freq` steps you will see evaluation results:
```
INFO ... eval/pc_success: 15.0   eval/avg_sum_reward: 23.5
```

### Success Criteria
- Training loss decreases over time
- `eval/pc_success` (success rate percentage) increases over time
- Checkpoint directory structure:
```
outputs/train/act_me5001_insertion/checkpoints/
├── 010000/
│   ├── pretrained_model/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── train_config.json
│   └── training_state/
├── 020000/
├── ...
└── last -> 050000
```

### Troubleshooting
- CUDA OOM: Reduce `--batch_size` (e.g., change to 4)
- Eval reports `ModuleNotFoundError: gym_aloha`: Go back to Step 2 and confirm you installed `".[aloha]"`
- To resume training after interruption:
  ```bash
  python lerobot/scripts/train.py \
    --config_path=outputs/train/act_me5001_insertion/checkpoints/last/pretrained_model/ \
    --resume=true
  ```

---

## 7. Inspect Checkpoint

### What to Do
Confirm checkpoint files are complete and can be loaded correctly.

### Commands to Run
```bash
# Check if files exist
ls -la outputs/train/act_me5001_insertion/checkpoints/last/pretrained_model/

# Verify the checkpoint can be loaded
python -c "
from lerobot.common.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained('outputs/train/act_me5001_insertion/checkpoints/last/pretrained_model')
print(f'Policy loaded, type={policy.config.type}')
print(f'Parameters: {sum(p.numel() for p in policy.parameters()):,}')
"
```

### Success Criteria
- All three files exist: `config.json`, `model.safetensors`, `train_config.json`
- Python loads without errors, outputting the policy type and parameter count

---

## 8. Simulation Environment Evaluation

### What to Do
Use the standalone eval script to evaluate the trained checkpoint in the `AlohaInsertion-v0` simulation.

### Why
Standalone evaluation can run more episodes to obtain a more accurate success rate.

### Commands to Run
```bash
python lerobot/scripts/eval.py \
  --policy.path=outputs/train/act_me5001_insertion/checkpoints/last/pretrained_model \
  --env.type=aloha \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --output_dir=outputs/eval/act_me5001_insertion \
  --seed=1000
```

**Parameter Explanation**:
- `--policy.path`: Points to the checkpoint directory
- `--eval.n_episodes=50`: Evaluate for 50 episodes
- `--eval.batch_size=10`: Run 10 environments in parallel

### Success Criteria
Terminal output similar to:
```
INFO ... eval/pc_success: XX.X   eval/avg_sum_reward: XX.X
```
Evaluation results are saved under `outputs/eval/act_me5001_insertion/`.

### Troubleshooting
- `FileNotFoundError`: Confirm `--policy.path` points to the correct checkpoint directory
- CUDA OOM: Reduce `--eval.batch_size`

---

## 9. Ablation Experiments (Optional)

### What to Do
Compare the effects of different datasets and hyperparameters on ACT to accumulate experimental data for the ME5001 paper.

### Experiment 1: Human vs Scripted Data
```bash
# Train with scripted dataset
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/aloha_sim_insertion_scripted \
  --policy.type=act \
  --env.type=aloha \
  --output_dir=outputs/train/act_insertion_scripted \
  --steps=50000 --eval_freq=10000 --policy.device=cuda
```
Compare `pc_success` with the human data results from Step 6.

### Experiment 2: ACT Hyperparameter Ablation
```bash
# Reduce chunk_size
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.type=act --env.type=aloha \
  --policy.chunk_size=50 \
  --output_dir=outputs/train/act_chunk50 \
  --steps=50000 --eval_freq=10000 --policy.device=cuda

# Disable VAE
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.type=act --env.type=aloha \
  --policy.use_vae=false \
  --output_dir=outputs/train/act_novae \
  --steps=50000 --eval_freq=10000 --policy.device=cuda
```

---

# Phase B: With Hardware

> The following steps require a connected Koch v1.1 robotic arm.

## 10. Koch Hardware Overview

### Koch v1.1 Components
- **Leader arm** (small arm, used for teleoperation): 6 Dynamixel XL330-M077 motors
- **Follower arm** (execution arm): 2 XL430-W250 (shoulder_pan, shoulder_lift) + 4 XL330-M288 (elbow_flex, wrist_flex, wrist_roll, gripper)
- Both arms connect to the computer via USB, one cable each
- Leader arm uses 5V power supply, follower arm uses 12V power supply (with a built-in voltage regulator to supply 5V to the XL330 motors)

### Default Koch Motor Configuration (defined in `KochRobotConfig` in `lerobot/common/robot_devices/robots/configs.py`)

**Leader arm (all xl330-m077)**:
| Motor Name | ID | Model |
|------------|-----|-------|
| shoulder_pan | 1 | xl330-m077 |
| shoulder_lift | 2 | xl330-m077 |
| elbow_flex | 3 | xl330-m077 |
| wrist_flex | 4 | xl330-m077 |
| wrist_roll | 5 | xl330-m077 |
| gripper | 6 | xl330-m077 |

**Follower arm**:
| Motor Name | ID | Model |
|------------|-----|-------|
| shoulder_pan | 1 | xl430-w250 |
| shoulder_lift | 2 | xl430-w250 |
| elbow_flex | 3 | xl330-m288 |
| wrist_flex | 4 | xl330-m288 |
| wrist_roll | 5 | xl330-m288 |
| gripper | 6 | xl330-m288 |

---

## 11. Dependency Installation (Dynamixel)

### What to Do
Install the Dynamixel SDK and keyboard listener library.

### Why
Koch uses Dynamixel servos. The dynamixel extra defined in `pyproject.toml` is:
```
dynamixel = ["dynamixel-sdk>=3.7.31", "pynput>=1.7.7"]
```

### Commands to Run
```bash
cd /home/kirito/ME5001
pip install -e ".[dynamixel]"
```

If you have already installed `".[aloha]"`, you can install both extras together:
```bash
pip install -e ".[dynamixel,aloha]"
```

### Success Criteria
```bash
python -c "import dynamixel_sdk; print('dynamixel_sdk OK')"
python -c "import pynput; print('pynput OK')"
```

---

## 12. Find Motor USB Ports

### What to Do
Determine the USB serial port paths for the leader arm and follower arm.

### Why
Port paths differ on each computer (Linux typically uses `/dev/ttyACM0`, `/dev/ttyACM1`; Mac typically uses `/dev/tty.usbmodemXXX`). You must find the correct ports for communication.

### Prerequisites
- Both arms are connected to the computer via USB
- Both arms are powered on (leader 5V, follower 12V)

### Commands to Run
Run twice to identify both arms:
```bash
python lerobot/scripts/find_motors_bus_port.py
```

**Interactive Process**:
1. The script displays all currently available ports
2. It prompts you to unplug one USB cable (e.g., the leader arm's)
3. Press Enter
4. The script identifies the disappeared port by comparison, which is that arm's port

**Repeat once** to identify the other arm.

### Linux Permission Issues
If you get a permission error:
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### Success Criteria
You obtain two port paths, for example:
- Leader arm: `/dev/ttyACM0`
- Follower arm: `/dev/ttyACM1`

Note down these two paths; they will be needed in subsequent steps.

---

## 13. Configure Motor IDs

### What to Do
Set a unique ID (1-6) for each motor. **This only needs to be done once when the motor is first used**; the ID is stored in the motor's permanent memory.

### Why
All motors ship with a default ID of 1. Motors on the same bus must have different IDs for proper communication.

### Prerequisites
- First disconnect all motors from the daisy-chain
- Connect only one motor to the bus at a time

### Commands to Run

**Configure Leader arm (all xl330-m077)**:

Connect and configure motors one by one. First connect motor 1:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM0 \
  --brand dynamixel \
  --model xl330-m077 \
  --baudrate 1000000 \
  --ID 1
```

Disconnect motor 1, connect motor 2:
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM0 \
  --brand dynamixel \
  --model xl330-m077 \
  --baudrate 1000000 \
  --ID 2
```

Repeat until ID 6. Six times total.

**Configure Follower arm**:

The first two motors are xl430-w250 (**note: use 12V power supply**):
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM1 \
  --brand dynamixel \
  --model xl430-w250 \
  --baudrate 1000000 \
  --ID 1
```

```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM1 \
  --brand dynamixel \
  --model xl430-w250 \
  --baudrate 1000000 \
  --ID 2
```

The last four motors are xl330-m288 (use 5V, through the voltage regulator):
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/ttyACM1 \
  --brand dynamixel \
  --model xl330-m288 \
  --baudrate 1000000 \
  --ID 3
```

Repeat until ID 6.

### After Configuration
Reconnect all motors in daisy-chain fashion.

### Success Criteria
All 12 motors (leader 6 + follower 6) are configured without errors.

### Troubleshooting
- `OSError: No motor found`: Check that the power is connected and USB is plugged in
- Multiple motors detected: Ensure only one motor is connected at a time
- When configuring XL430, make sure to use 12V power supply

---

## 14. Update KochRobotConfig Ports

### What to Do
Write your actual USB ports into the Koch robot configuration.

### Why
The default ports in `KochRobotConfig` are `/dev/tty.usbmodem585A0085511` (Mac format). You need to change them to the actual ports found in Step 12.

### Commands to Run
Edit `lerobot/common/robot_devices/robots/configs.py`, find `class KochRobotConfig` (around line 211), and modify the `port` fields:

```python
@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyACM0",  # <- Change to your leader arm port
                motors={
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/ttyACM1",  # <- Change to your follower arm port
                motors={
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )
    # Leave cameras unchanged for now; configure in Step 17
```

Alternatively, you can override ports via command-line arguments without modifying source code (see Step 16).

### Success Criteria
File is modified and saved without syntax errors.

---

## 15. Calibrate the Robotic Arms

### What to Do
Calibrate the leader arm and follower arm so that both arms output the same position values when at the same physical position.

### Why
Different motors may have different zero points and rotation directions. Calibration ensures:
1. Position values of both arms are physically aligned
2. A policy trained on one Koch can be transferred to another

### Calibration Only Needs to Be Done Once
Calibration data is saved in `.cache/calibration/koch/main_follower.json` and `.cache/calibration/koch/main_leader.json`.

### Commands to Run
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --robot.cameras='{}' \
  --control.type=calibrate
```

`--robot.cameras='{}'` skips the camera (cameras are not yet configured at this point).

### Calibration Process
The script will guide you to move each arm to three positions:

1. **Zero position**: Arm straight, horizontal, gripper facing up and closed
2. **Rotated position**: All joints rotated approximately 90 degrees
3. **Rest position**: Arm placed in a natural resting position

Press Enter to confirm after reaching each position.

### Success Criteria
```bash
ls .cache/calibration/koch/
# Should see:
# main_follower.json
# main_leader.json
```

### Troubleshooting
- Before calibration, all motor torques must be disabled; the arms should move freely
- If you get `OSError`: Check if the ports are correct (Step 14)
- If calibration results are abnormal (arm jumps suddenly during teleop): Delete `.cache/calibration/koch/` and recalibrate

---

## 16. Teleoperation Test

### What to Do
Control the follower arm in real time by moving the leader arm to verify the hardware link is working.

### Commands to Run
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --robot.cameras='{}' \
  --control.type=teleoperate
```

If you did not modify the ports in the source code, you can also override them via command line:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --robot.cameras='{}' \
  --robot.leader_arms.main.port=/dev/ttyACM0 \
  --robot.follower_arms.main.port=/dev/ttyACM1 \
  --control.type=teleoperate
```

### Runtime Logs
```
dt: 5.12 (195.1hz) dtRlead: 4.93 (203.0hz) dtWfoll: 0.19 (5239.0hz)
```
- `dt`: Total time for one teleop step
- `dtRlead`: Time to read the leader arm position
- `dtWfoll`: Time to write the target position to the follower arm

### Success Criteria
When moving the leader arm, the follower arm follows in real time with no noticeable delay or jitter.

### Troubleshooting
- Arm does not move: Check that the power is connected
- Arm jerks violently: Calibration may be incorrect; redo Step 15
- Press `Ctrl+C` to end teleoperation

---

## 17. Configure Cameras

### What to Do
Find camera device indices and configure them in the Koch robot.

### Why
The ACT policy requires image input. Cameras are the source of `observation.images.{camera_name}`.

### Find Camera Indices
```bash
python lerobot/common/robot_devices/cameras/opencv.py \
  --images-dir outputs/images_from_opencv_cameras
```

The script scans all available cameras and saves a few frames to `outputs/images_from_opencv_cameras/`. Review the images to determine which index corresponds to which physical camera.

### Update Camera Configuration in KochRobotConfig
Edit the `cameras` field of `KochRobotConfig` in `lerobot/common/robot_devices/robots/configs.py`.

For example, if you have a USB webcam (index 0) and a phone camera (index 1):
```python
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "laptop": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
        "phone": OpenCVCameraConfig(
            camera_index=1,
            fps=30,
            width=640,
            height=480,
        ),
    }
)
```

If you only have one camera:
```python
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "webcam": OpenCVCameraConfig(
            camera_index=0,
            fps=30,
            width=640,
            height=480,
        ),
    }
)
```

### Verify Camera + Teleoperation
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=teleoperate
```
A camera feed window should now pop up.

### Success Criteria
- Teleoperation works normally
- Camera feed displays in real time
- If you cannot see the camera window but teleoperation works fine: Check the `$DISPLAY` environment variable (Linux)

---

## 18. Record Dataset

### What to Do
Record training data via teleoperation and save it in LeRobot dataset format.

### Why
This is the data source for imitation learning. ACT requires `observation.state` (joint positions), `observation.images.*` (images), and `action` (target joint positions).

### Set Up HuggingFace Account (if uploading)
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

### Commands to Run
Record 2 test episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.single_task="Pick up a lego block and place it in the bin." \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=false
```

**Parameter Explanation**:
- `--control.single_task`: Task description text, stored in dataset metadata
- `--control.fps=30`: Recording frame rate of 30Hz
- `--control.warmup_time_s=5`: 5-second warmup before recording (hardware synchronization)
- `--control.episode_time_s=30`: Record 30 seconds per episode
- `--control.reset_time_s=30`: 30 seconds between episodes to reset the environment
- `--control.num_episodes=2`: Record 2 episodes
- `--control.push_to_hub=false`: Do not upload to HuggingFace for now

### Keyboard Controls During Recording
- `->` (Right arrow): End the current episode early and enter reset
- `<-` (Left arrow): Cancel the current episode and re-record
- `ESC`: End the entire recording session early

### Full Recording (50+ episodes)
After testing is successful, record the complete dataset:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.single_task="Pick up a lego block and place it in the bin." \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_pick_place_lego \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=50 \
  --control.push_to_hub=true
```

### Recording Tips (from tutorial section 3b)
- At least 50 episodes
- Keep the camera position fixed
- Keep the manipulation behavior consistent
- You can record 10 episodes at each of 10 different object positions
- Do not add too much variation at the beginning

### Success Criteria
Dataset is saved at `~/.cache/huggingface/lerobot/${HF_USER}/koch_test/`, containing:
- `meta/info.json`
- `data/chunk-000/episode_000000.parquet`
- `videos/chunk-000/observation.images.*/episode_000000.mp4`

### Troubleshooting
- Camera feed is choppy: On Linux, try `pip uninstall opencv-python && conda install -c conda-forge opencv=4.10.0`
- Video encoding error `ffmpeg: unknown encoder libsvtav1`: `conda install -c conda-forge ffmpeg`
- To resume after recording interruption: Add `--control.resume=true` at the end of the command
- Arrow keys/ESC not working on Linux: Check the `$DISPLAY` environment variable

---

## 19. Inspect Recorded Data

### What to Do
Visualize the recorded dataset to confirm data quality.

### Commands to Run
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test
```

If the dataset was not uploaded to HuggingFace Hub:
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test \
  --load-from-hf-hub 0
```

This starts a local web server (default `http://127.0.0.1:9090`) where you can view the video and state curves for each episode in a browser.

### Success Criteria
You can see the recorded videos and joint state trajectories in the browser.

---

## 20. Replay Verification

### What to Do
Replay a recorded episode on the robot to verify the accuracy and reproducibility of the data.

### Commands to Run
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.episode=0
```

### Success Criteria
The follower arm reproduces the motion trajectory from the recording.

---

## 21. Train ACT with Your Own Data

### What to Do
Train an ACT policy using the real data recorded in Step 18.

### Commands to Run
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_pick_place_lego \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_pick_place_lego \
  --job_name=act_koch_pick_place_lego \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=20000 \
  --policy.device=cuda \
  --wandb.enable=true
```

**Notes**:
- **Do not set `--env.type`** here, because there is no corresponding simulation environment. Training only does offline learning without eval rollouts
- `--wandb.enable=true` is optional; use it to track training curves on Weights & Biases (requires running `wandb login` first)
- The ACT policy will automatically adapt to the number of motors and cameras in your dataset (read from dataset metadata)

### Success Criteria
- Loss decreases continuously
- Checkpoints are saved at `outputs/train/act_koch_pick_place_lego/checkpoints/`

### Upload Checkpoint (Optional)
```bash
huggingface-cli upload ${HF_USER}/act_koch_pick_place_lego \
  outputs/train/act_koch_pick_place_lego/checkpoints/last/pretrained_model
```

---

## 22. Real Robot Deployment and Evaluation

### What to Do
Use the trained ACT policy to control the Koch robot to autonomously execute tasks, and record evaluation data.

### Commands to Run
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up a lego block and place it in the bin." \
  --control.repo_id=${HF_USER}/eval_act_koch_pick_place_lego \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_koch_pick_place_lego/checkpoints/last/pretrained_model
```

**Differences from recording training data**:
- Added `--control.policy.path`: Uses a neural network to provide actions instead of human teleoperation
- `repo_id` is prefixed with `eval_` for distinction

### View After Evaluation
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/eval_act_koch_pick_place_lego
```

### Success Criteria
- The robot can execute actions autonomously
- The video shows task completion (or partial completion)
- Calculate success rate through multiple evaluations

---

# Phase C: Migrate to ME5001 Insertion Task

> The following describes how to migrate the above Koch + pick-and-place workflow to the ME5001 insertion task.

## Differences from Pick-and-Place

| Dimension | Pick-and-place | Battery/USB Insertion |
|-----------|---------------|----------------------|
| Precision requirement | Medium (cm level) | High (mm level) |
| Visual feedback | Optional | Required (closed-loop) |
| Typical episode duration | 20-30 seconds | 10-20 seconds |
| Key camera position | Top-down view is sufficient | Needs wrist cam or close-up side view |
| Recommended number of episodes | 50+ | 50+, multiple angles |

## Migration Steps

### C1. Task Definition
Modify `--control.single_task` to your task description:
```
"Insert a battery into the battery slot."
"Insert a USB-A plug into the USB port."
```

### C2. Camera Configuration Adjustment
Insertion tasks require close-up visual feedback. Recommendations:
- Add a wrist camera or close-up fixed camera
- Keep resolution at 640x480, fps 30
- Update the `cameras` field in `KochRobotConfig`

### C3. Recording Parameter Adjustment
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.single_task="Insert a battery into the battery slot." \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/me5001_battery_insertion \
  --control.warmup_time_s=5 \
  --control.episode_time_s=20 \
  --control.reset_time_s=15 \
  --control.num_episodes=50 \
  --control.push_to_hub=false
```

Changes:
- `episode_time_s=20`: Insertion actions are shorter
- `reset_time_s=15`: Faster reset
- `single_task` changed to your task description

### C4. Training Command
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/me5001_battery_insertion \
  --policy.type=act \
  --output_dir=outputs/train/act_me5001_battery \
  --job_name=act_me5001_battery \
  --batch_size=8 \
  --steps=100000 \
  --save_freq=20000 \
  --policy.device=cuda
```

### C5. ACT Hyperparameter Recommendations
For fine manipulation tasks like insertion, you may need to adjust:
- `--policy.chunk_size=50`: Reduce chunk size for finer control
- `--policy.kl_weight=10.0`: Default value, controls the regularization strength of the VAE

The optimal values need to be determined through experiments.

### C6. Deployment Evaluation
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Insert a battery into the battery slot." \
  --control.repo_id=${HF_USER}/eval_act_me5001_battery \
  --control.episode_time_s=20 \
  --control.num_episodes=10 \
  --control.policy.path=outputs/train/act_me5001_battery/checkpoints/last/pretrained_model
```

### C7. Evaluation Metrics
You need to manually record:
- **Success rate**: Whether the battery is fully inserted
- **Completion time**: Time from start to insertion completion
- **Number of interventions**: Whether manual intervention was needed

---

# Appendix: Common Errors and Fixes

### Installation Phase

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'lerobot'` | Not installed | `cd /home/kirito/ME5001 && pip install -e .` |
| `ModuleNotFoundError: No module named 'gym_aloha'` | aloha extra not installed | `pip install -e ".[aloha]"` |
| `ModuleNotFoundError: No module named 'dynamixel_sdk'` | dynamixel extra not installed | `pip install -e ".[dynamixel]"` |

### Training Phase

| Error | Cause | Fix |
|-------|-------|-----|
| `FileExistsError: Output directory already exists` | output_dir already exists | Delete the directory or use a different `--output_dir` |
| `torch.cuda.OutOfMemoryError` | Insufficient GPU memory | Reduce `--batch_size` (e.g., to 4 or 2) |
| Network download timeout | HuggingFace Hub access issue | Set `HF_ENDPOINT=https://hf-mirror.com` (China mirror) |
| Training interrupted | Any reason | `--config_path=<checkpoint_dir>/pretrained_model/ --resume=true` |

### Hardware Phase

| Error | Cause | Fix |
|-------|-------|-----|
| `OSError: No motor found` | Power off / USB disconnected | Check power and USB connections |
| `PermissionError: /dev/ttyACM0` | Linux serial port permissions | `sudo chmod 666 /dev/ttyACM0` |
| Arm jerks violently | Calibration error | Delete `.cache/calibration/koch/` and recalibrate |
| `ffmpeg: unknown encoder libsvtav1` | ffmpeg missing encoder | `conda install -c conda-forge ffmpeg` |
| Camera shows black/green screen | Camera warming up | Wait a few seconds; first few frames may be abnormal |
| Arrow keys not working on Linux | pynput requires DISPLAY | Set `export DISPLAY=:0` |

### Checkpoint Phase

| Error | Cause | Fix |
|-------|-------|-----|
| `FileNotFoundError: config.json` | Incorrect checkpoint path | Confirm the path points to the `pretrained_model/` directory |
| Inference results are all wrong after loading | Insufficient training/data | Increase `--steps` or record more episodes |
