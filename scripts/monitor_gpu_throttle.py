#!/usr/bin/env python3
"""
monitor_gpu_throttle.py -- log GPU thermal/power throttle state + what's
running on each GPU, over time, to correlate throttling events with which
script/job triggered them.

Polls nvidia-smi's own throttle-reason flags directly (not just raw
temperature, which only tells you it's hot, not that anything is actually
being slowed down) -- these are the same values nvidia-smi reports for
"why are my clocks lower than requested":
  - hw_thermal_slowdown / sw_thermal_slowdown: temperature-driven
  - sw_power_cap / hw_power_brake_slowdown: power-driven
Also records current vs. max SM clock (the direct, quantitative measure of
how much throttling is actually costing you), and resolves the full
command line of whatever process is using each GPU at poll time, so a
later join between "GPU 1 started throttling at 14:32" and "what was
running" is a straight timestamp lookup, not a guess.

Usage:
    python3 monitor_gpu_throttle.py --output gpu_throttle_log.csv --interval 30

Runs until killed (Ctrl-C or `kill`). Safe to leave running continuously
alongside real jobs -- one nvidia-smi call per poll, negligible overhead.
"""
import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

GPU_FIELDS = [
    'index', 'temperature.gpu', 'utilization.gpu', 'power.draw', 'power.limit',
    'clocks.current.sm', 'clocks.max.sm',
    'clocks_throttle_reasons.active',
    'clocks_throttle_reasons.hw_thermal_slowdown',
    'clocks_throttle_reasons.sw_thermal_slowdown',
    'clocks_throttle_reasons.sw_power_cap',
    'clocks_throttle_reasons.hw_power_brake_slowdown',
]

CSV_HEADER = [
    'timestamp', 'gpu_index', 'temp_c', 'util_pct', 'power_w', 'power_limit_w',
    'clock_sm_mhz', 'clock_max_sm_mhz', 'clock_pct_of_max',
    'throttle_active', 'throttle_hw_thermal', 'throttle_sw_thermal',
    'throttle_sw_power_cap', 'throttle_hw_power_brake',
    'pid', 'gpu_mem_mib', 'command_short', 'command_full',
]


def _query_gpus():
    out = subprocess.run(
        ['nvidia-smi', f'--query-gpu={",".join(GPU_FIELDS)}',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        rows.append(dict(zip(
            ['index', 'temp', 'util', 'power', 'power_limit', 'clk_sm', 'clk_max',
             'thr_active', 'thr_hw_therm', 'thr_sw_therm', 'thr_sw_pwr', 'thr_hw_pwr'],
            parts,
        )))
    return rows


def _query_compute_procs():
    """pid -> (gpu_uuid, used_memory_mib)"""
    out = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True,
    ).stdout
    procs = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, uuid, mem = [p.strip() for p in line.split(',')]
        procs[pid] = (uuid, mem)
    return procs


def _gpu_index_by_uuid():
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
        capture_output=True, text=True, check=True,
    ).stdout
    m = {}
    for line in out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(',')]
        m[uuid] = idx
    return m


def _short_command(cmdline):
    """Pull out something readable (a TS name if present) from a long
    pytom_match_template.py / pytom_extract_candidates.py invocation."""
    m = re.search(r'(ts-\d+)', cmdline)
    if m:
        prog = cmdline.split()[0].rsplit('/', 1)[-1] if cmdline else ''
        return f'{prog} {m.group(1)}'
    return cmdline.split()[0].rsplit('/', 1)[-1] if cmdline else ''


def _cmdline_for_pid(pid):
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
        return raw.replace(b'\0', b' ').decode(errors='replace').strip()
    except (FileNotFoundError, PermissionError):
        return ''


def poll_once():
    gpus = _query_gpus()
    procs = _query_compute_procs()  # pid -> (uuid, mem)
    idx_by_uuid = _gpu_index_by_uuid()
    # invert: gpu_index -> [(pid, mem), ...]
    by_index = {}
    for pid, (uuid, mem) in procs.items():
        idx = idx_by_uuid.get(uuid)
        if idx is not None:
            by_index.setdefault(idx, []).append((pid, mem))

    ts = datetime.now().isoformat(timespec='seconds')
    rows = []
    for g in gpus:
        entries = by_index.get(g['index'], [(None, None)])
        for pid, mem in entries:
            cmd_full = _cmdline_for_pid(pid) if pid else ''
            clk_sm = float(g['clk_sm']) if g['clk_sm'] not in ('', '[N/A]') else None
            clk_max = float(g['clk_max']) if g['clk_max'] not in ('', '[N/A]') else None
            pct = f'{100*clk_sm/clk_max:.0f}' if clk_sm and clk_max else ''
            rows.append([
                ts, g['index'], g['temp'], g['util'], g['power'], g['power_limit'],
                g['clk_sm'], g['clk_max'], pct,
                g['thr_active'], g['thr_hw_therm'], g['thr_sw_therm'],
                g['thr_sw_pwr'], g['thr_hw_pwr'],
                pid or '', mem or '', _short_command(cmd_full), cmd_full,
            ])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', '-o', default='gpu_throttle_log.csv')
    ap.add_argument('--interval', type=float, default=30.0, help='seconds between polls')
    args = ap.parse_args()

    out_path = Path(args.output)
    write_header = not out_path.exists()
    print(f'Logging every {args.interval:.0f}s to {out_path.resolve()} (Ctrl-C to stop)')

    with open(out_path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CSV_HEADER)
            fh.flush()
        try:
            while True:
                t0 = time.time()
                try:
                    for row in poll_once():
                        writer.writerow(row)
                    fh.flush()
                except subprocess.CalledProcessError as e:
                    print(f'WARNING: nvidia-smi call failed: {e}', file=sys.stderr)
                elapsed = time.time() - t0
                time.sleep(max(0.0, args.interval - elapsed))
        except KeyboardInterrupt:
            print('\nStopped.')


if __name__ == '__main__':
    main()
