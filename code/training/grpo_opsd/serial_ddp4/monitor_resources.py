"""Sample physical GPUs 1/2/3 while the bound benchmark PID is live."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def alive(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():return False
    assert p.stat().st_uid == os.getuid()
    return p.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def main(pid, output, max_seconds=14500):
    output = Path(output)
    report = {'pid': pid, 'status': 'sampling', 'interval_seconds': 15, 'samples': 0,
              'gpu': {str(i): {'maximum_used_mib': 0, 'minimum_free_mib': 100000,
                               'maximum_utilization_percent': 0} for i in (0, 1, 2, 3)},
              'observed_compute_processes': {}}
    deadline = time.monotonic() + max_seconds
    while alive(pid) and time.monotonic() < deadline:
        data = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        ids = {}
        for line in data.splitlines():
            index, uuid, used, free, utilization = [x.strip() for x in line.split(',')]
            if index not in report['gpu']:continue
            ids[uuid] = index
            row = report['gpu'][index]
            row['maximum_used_mib'] = max(row['maximum_used_mib'], int(used))
            row['minimum_free_mib'] = min(row['minimum_free_mib'], int(free))
            row['maximum_utilization_percent'] = max(row['maximum_utilization_percent'], int(utilization))
        data = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                        '--format=csv,noheader,nounits'], text=True, timeout=10)
        for line in data.splitlines():
            process, uuid, memory = [x.strip() for x in line.split(',')]
            if uuid not in ids:continue
            p = Path('/proc') / process
            try:
                owned = p.stat().st_uid == os.getuid()
                command = (p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
                           if owned else '[external process]')
            except FileNotFoundError:continue
            key = ids[uuid] + '/' + process
            row = report['observed_compute_processes'].setdefault(key, {'gpu_index': int(ids[uuid]),
                          'pid': int(process), 'owned_by_training_user': owned,
                          'command': command, 'maximum_used_mib': 0})
            row['maximum_used_mib'] = max(row['maximum_used_mib'], int(memory))
        report['samples'] += 1;report['last_sample_time'] = time.time()
        tmp = output.with_suffix('.tmp');tmp.write_text(json.dumps(report, indent=2) + '\n');tmp.replace(output)
        time.sleep(15)
    report['status'] = 'completed' if not alive(pid) else 'sampling_deadline_reached'
    tmp = output.with_suffix('.tmp');tmp.write_text(json.dumps(report, indent=2) + '\n');tmp.replace(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--pid', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-seconds', type=int, default=14500)
    a = p.parse_args();assert a.max_seconds > 0;main(a.pid, a.output, a.max_seconds)
