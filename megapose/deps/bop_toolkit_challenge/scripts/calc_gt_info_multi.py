import subprocess
import time
import numpy as np

if __name__ == '__main__':
    start_scene = 22
    end_scene = 49
    n_procs = 4
    scene_ids = np.arange(start_scene, end_scene + 1)
    scene_ids_split = np.array_split(scene_ids, n_procs)

    processes = []
    for this_scene_ids in scene_ids_split:
        cmd = [
            'python',
            '/home/ylabbe/projects/bop_toolkit_challenge/scripts/calc_gt_info.py',
            '--start_scene', str(min(this_scene_ids)),
            '--end_scene', str(max(this_scene_ids)),
        ]
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    while True:
        is_done = [process.poll() is not None for process in processes]
        if all(is_done):
            break
        time.sleep(.5)
