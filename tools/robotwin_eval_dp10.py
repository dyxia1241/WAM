#!/usr/bin/env python3
"""Run RoboTwin DP evaluation for a fixed small number of rollouts.

This is intended to be executed from the RoboTwin repository root on 5060.
It reuses RoboTwin's official script/eval_policy.py helpers, but calls
eval_policy with test_num=10 instead of the official script default of 100.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")

import script.eval_policy as ep
from envs import CONFIGS_PATH
from generate_episode_instructions import generate_episode_descriptions  # noqa: F401


def get_embodiment_config(robot_file):
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="click_bell")
    parser.add_argument("--task_config", default="ppwam_dp10_clean")
    parser.add_argument("--ckpt_setting", default="ppwam_dp10_clean")
    parser.add_argument("--expert_data_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_num", default="600")
    parser.add_argument("--policy_name", default="DP")
    parser.add_argument("--instruction_type", default="unseen")
    parser.add_argument("--test_num", type=int, default=10)
    args_ns = parser.parse_args()

    usr_args = vars(args_ns)
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"No embodiment file for {name}")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["policy_name"] = policy_name

    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}_test{args_ns.test_num}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_size = None
    if args["eval_video_log"]:
        camera_cfg = ep.get_camera_config(args["camera"]["head_camera_type"])
        video_size = f"{camera_cfg['w']}x{camera_cfg['h']}"
        args["eval_video_save_dir"] = save_dir

    get_model = ep.eval_function_decorator(policy_name, "get_model")
    model = get_model(usr_args)
    task_env = ep.class_decorator(task_name)
    st_seed = 100000 * (1 + args_ns.seed)

    _, suc_num = ep.eval_policy(
        task_name,
        task_env,
        args,
        model,
        st_seed,
        test_num=args_ns.test_num,
        video_size=video_size,
        instruction_type=args_ns.instruction_type,
    )

    result_path = save_dir / "_result.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Policy: {policy_name}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Checkpoint: {ckpt_setting}/{args_ns.checkpoint_num}\n")
        f.write(f"Success: {suc_num}/{args_ns.test_num}\n")
        f.write(f"Success rate: {suc_num / args_ns.test_num:.4f}\n")

    print(f"DP eval10 saved to {save_dir}")
    print(f"Success: {suc_num}/{args_ns.test_num} ({np.round(suc_num / args_ns.test_num * 100, 1)}%)")


if __name__ == "__main__":
    main()
