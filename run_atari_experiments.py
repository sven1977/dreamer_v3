# Runs n Atari100k experiments.

import os
import re
import subprocess

envs = [
    "Boxing",
    "Hero",
    "Kangaroo",
    "Krull",
    "MsPacman",
    "Pong",
    "Qbert",
    "Roadrunner",
]

# Kill all running experiments.
y = input("Kill running jobs?\n[y|n]>")
if y.lower().startswith("y"):
    running = subprocess.run(["ps", "-ef"], capture_output=True)
    for line in running.stdout.decode("utf-8"):
        mo = re.match("^\\w+\\s+(\\d+)", line)
        if mo:
            job_id = mo.groups(1)
            os.system(f"kill -9 {job_id}")

for cuda_id, env in enumerate(envs):
    print(f"Starting Atari100k {env} ...")
    os.system(
        f"CUDA_VISIBLE_DEVICES={cuda_id} python run_experiment.py "
        f"-c examples/atari_100k.yaml -t ALE/{env}-v5 > out_{env.lower()}.txt 2>&1 &"
    )

print("ok")
