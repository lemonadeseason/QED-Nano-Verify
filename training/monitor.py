#!/usr/bin/env python3
"""RL Training Monitor — shows real-time throughput for each pipeline stage."""
import re, os, sys, time, glob
from pathlib import Path
from datetime import datetime

def find_latest_run():
    runs = sorted(glob.glob("results/rl-8gpu-*/"), reverse=True)
    return runs[0] if runs else None

def parse_ts(line):
    """Extract datetime from actor or finetune log line."""
    # Actor format: 2026-04-28 06:12:06,951
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if m:
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    # Finetune format: 04/28/2026 06:10:03.421
    m = re.search(r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})', line)
    if m:
        return datetime.strptime(m.group(1), '%m/%d/%Y %H:%M:%S')
    return None

def analyze_rollouts(run_dir):
    log = Path(run_dir) / "actor/info.log"
    if not log.exists():
        return {}

    lines = log.read_text().splitlines()
    first_ts = None
    milestones = []  # (timestamp, finished_count)

    for line in lines:
        m = re.search(r'finished so far: (\d+)', line)
        if m:
            n = int(m.group(1))
            ts = parse_ts(line)
            if ts:
                if first_ts is None:
                    first_ts = ts
                milestones.append((ts, n))

    if not milestones:
        return {"rollouts_finished": 0}

    last_ts, last_n = milestones[-1]
    # Find first non-zero
    first_nonzero = None
    for ts, n in milestones:
        if n > 0 and first_nonzero is None:
            first_nonzero = (ts, n)
            break

    result = {
        "rollouts_finished": last_n,
        "rollouts_started": 0,
        "elapsed_min": (last_ts - first_ts).total_seconds() / 60 if first_ts else 0,
    }

    # Extract started count
    for line in reversed(lines):
        m = re.search(r'started so far: (\d+)', line)
        if m:
            result["rollouts_started"] = int(m.group(1))
            break

    if first_nonzero and last_n > first_nonzero[1]:
        dt = (last_ts - first_nonzero[0]).total_seconds() / 60
        dn = last_n - first_nonzero[1]
        result["rollout_rate_per_min"] = dn / dt if dt > 0 else 0
        result["avg_rollout_time_min"] = 0
        # Estimate: 96 concurrent / rate = avg time per rollout
        result["avg_rollout_time_min"] = 96 / result["rollout_rate_per_min"] if result["rollout_rate_per_min"] > 0 else 0
    elif first_nonzero:
        result["rollout_rate_per_min"] = 0
    else:
        result["rollout_rate_per_min"] = 0

    return result

def analyze_grader(run_dir):
    """Analyze grader (GPT-5.2 / gpt-oss-20b) call timings from actor debug log.
    
    Pairs up 'Sending HTTP Request' and 'HTTP Response' lines for gpt-5-2 calls
    to compute per-request latency. Also extracts rate limit info from headers.
    """
    log = Path(run_dir) / "actor/debug.log"
    if not log.exists():
        return {}

    result = {}
    send_times = []  # timestamps of Sending HTTP Request
    recv_times = []  # timestamps of HTTP Response 200
    latencies = []
    response_sizes = []
    rate_limit_remaining_tokens = None
    rate_limit_limit_tokens = None
    errors = 0

    for line in log.open():
        # Match grader request send
        if "Sending HTTP Request" in line and "gpt-5-2" in line:
            ts = parse_ts(line)
            if ts:
                send_times.append(ts)

        # Match grader response
        if "HTTP Response" in line and "gpt-5-2" in line:
            ts = parse_ts(line)
            if ts:
                recv_times.append(ts)
            # Extract response size
            m_size = re.search(r"content-length.*?'(\d+)'", line)
            if m_size:
                response_sizes.append(int(m_size.group(1)))
            # Extract rate limit info (latest)
            m_remaining = re.search(r"x-ratelimit-remaining-tokens.*?'(\d+)'", line)
            if m_remaining:
                rate_limit_remaining_tokens = int(m_remaining.group(1))
            m_limit = re.search(r"x-ratelimit-limit-tokens.*?'(\d+)'", line)
            if m_limit:
                rate_limit_limit_tokens = int(m_limit.group(1))
            # Check for non-200
            if "200 OK" not in line:
                errors += 1

    # Pair send/recv to compute latencies
    # They should be in order since the log is sequential per-thread
    for i in range(min(len(send_times), len(recv_times))):
        dt = (recv_times[i] - send_times[i]).total_seconds()
        if 0 < dt < 600:  # sanity: ignore if > 10 min or negative
            latencies.append(dt)

    result["total_grading_calls"] = len(recv_times)
    result["errors"] = errors

    # Score distribution from info.log
    info_log = Path(run_dir) / "actor/info.log"
    if info_log.exists():
        score_counts = {}
        for line in info_log.open():
            m = re.search(r'Parsed score=(\d+)', line)
            if m:
                s = int(m.group(1))
                score_counts[s] = score_counts.get(s, 0) + 1
        if score_counts:
            result["score_distribution"] = score_counts
            result["total_scored"] = sum(score_counts.values())
            result["nonzero_scored"] = sum(v for k, v in score_counts.items() if k > 0)

    if latencies:
        result["avg_latency_sec"] = sum(latencies) / len(latencies)
        result["min_latency_sec"] = min(latencies)
        result["max_latency_sec"] = max(latencies)
        result["median_latency_sec"] = sorted(latencies)[len(latencies) // 2]
        result["p90_latency_sec"] = sorted(latencies)[int(len(latencies) * 0.9)]

    if response_sizes:
        result["avg_response_bytes"] = sum(response_sizes) / len(response_sizes)

    if recv_times and len(recv_times) >= 2:
        total_time = (recv_times[-1] - recv_times[0]).total_seconds() / 60
        if total_time > 0:
            result["grading_rate_per_min"] = (len(recv_times) - 1) / total_time

    if rate_limit_limit_tokens:
        result["rate_limit_tokens"] = rate_limit_limit_tokens
        result["rate_limit_remaining_tokens"] = rate_limit_remaining_tokens
        if rate_limit_remaining_tokens is not None:
            result["rate_limit_usage_pct"] = (1 - rate_limit_remaining_tokens / rate_limit_limit_tokens) * 100

    return result

def analyze_preprocess(run_dir):
    log = Path(run_dir) / "preprocess/debug.log"
    info_log = Path(run_dir) / "preprocess/info.log"
    result = {"samples_ready": 0, "target": 512}

    if log.exists():
        lines = log.read_text().splitlines()
        for line in reversed(lines):
            m = re.search(r'trainer 0 has (\d+) samples', line)
            if m:
                result["samples_ready"] = int(m.group(1))
                result["last_preprocess_ts"] = parse_ts(line)
                break

        # Find first sample timestamp
        for line in lines:
            m = re.search(r'trainer 0 has (\d+) samples', line)
            if m and int(m.group(1)) > 0:
                result["first_sample_ts"] = parse_ts(line)
                break

        if result.get("first_sample_ts") and result.get("last_preprocess_ts") and result["samples_ready"] > 0:
            dt = (result["last_preprocess_ts"] - result["first_sample_ts"]).total_seconds() / 60
            result["sample_rate_per_min"] = result["samples_ready"] / dt if dt > 0 else 0
            result["est_minutes_to_512"] = (512 - result["samples_ready"]) / result["sample_rate_per_min"] if result["sample_rate_per_min"] > 0 else float('inf')

    if info_log.exists():
        text = info_log.read_text()
        processed = re.findall(r'Processed (\d+) samples.*?in ([\d.]+)s', text)
        if processed:
            total_samples = sum(int(x[0]) for x in processed)
            total_time = sum(float(x[1]) for x in processed)
            result["preprocess_batches"] = len(processed)
            result["total_processed"] = total_samples
            result["avg_preprocess_sec_per_sample"] = total_time / total_samples if total_samples > 0 else 0

    return result

def analyze_finetune(run_dir):
    log = Path(run_dir) / "finetune/log/info_0.log"
    if not log.exists():
        return {}

    text = log.read_text()
    result = {}

    # Extract step start/end timestamps
    start_times = re.findall(r"Start step at ([\d.]+)", text)
    
    # Find completed steps - parse the dict-like string robustly
    completed_matches = list(re.finditer(r"Completed steps (\d+): \{(.+?)\}", text))

    if completed_matches:
        all_steps = []
        for cm in completed_matches:
            step_num = int(cm.group(1))
            stats_str = cm.group(2)
            # Parse key-value pairs from the stats string
            def get_val(key, s):
                m = re.search(rf"'{key}':\s*'?([^',}}]+)'?", s)
                return float(m.group(1)) if m else 0.0
            
            step_data = {
                "step": step_num,
                "reward": get_val("rl/reward", stats_str),
                "loss": get_val("rl/loss", stats_str),
                "sec_per_pass": get_val("throughput/sec_per_pass", stats_str),
                "time_waiting": get_val("stats/time_waiting_for_data", stats_str),
                "sec_per_step": get_val("throughput/sec_per_step", stats_str),
                "passes": get_val("stats/passes", stats_str),
                "grad_norm": get_val("stats/grad_norm", stats_str),
                "samples": get_val("stats/samples", stats_str),
            }
            all_steps.append(step_data)

        latest = all_steps[-1]
        result["completed_steps"] = latest["step"]
        result["latest_reward"] = latest["reward"]
        result["latest_loss"] = latest["loss"]
        result["latest_sec_per_pass"] = latest["sec_per_pass"]
        result["latest_time_waiting"] = latest["time_waiting"]
        result["latest_sec_per_step"] = latest["sec_per_step"]
        result["latest_grad_norm"] = latest["grad_norm"]
        result["latest_passes"] = latest["passes"]

        result["reward_history"] = [s["reward"] for s in all_steps]
        result["avg_sec_per_pass"] = sum(s["sec_per_pass"] for s in all_steps) / len(all_steps)
        result["avg_sec_per_step"] = sum(s["sec_per_step"] for s in all_steps) / len(all_steps)
        result["all_steps"] = all_steps

    # Determine current step status even if no step completed yet
    # Count timeout resets (0.1) = number of gradient accumulation passes consumed
    timeout_resets = text.count("retrying with timeout 0.1\n")
    # First reset is from initial startup, subtract 1 per step start
    num_step_starts = len(start_times)
    passes_done = max(0, timeout_resets - num_step_starts)
    
    if not completed_matches:
        if "Batch queue is empty" in text:
            result["status"] = f"step 1 in progress ({passes_done} passes done)"
        else:
            result["status"] = "initializing"
    else:
        # There's a new step in progress after the last completed one
        current_step = result["completed_steps"] + 1
        # Count resets after the last completed step
        last_complete_pos = completed_matches[-1].end()
        remaining_text = text[last_complete_pos:]
        current_passes = remaining_text.count("retrying with timeout 0.1\n")
        # Subtract 1 for the step-start reset
        current_passes = max(0, current_passes - 1)
        result["current_step"] = current_step
        result["current_passes"] = current_passes

    return result

def analyze_gpu():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            text=True
        )
        gpus = []
        for line in out.strip().split('\n'):
            parts = [p.strip().replace(' MiB', '').replace(' %', '') for p in line.split(',')]
            gpus.append({
                "idx": int(parts[0]),
                "util": int(parts[1]),
                "mem_used": int(parts[2]),
                "mem_total": int(parts[3]),
            })
        return gpus
    except Exception:
        return []

def print_report(run_dir):
    run_name = os.path.basename(run_dir.rstrip('/'))
    print(f"\n{'='*70}")
    print(f"  RL Training Monitor — {run_name}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # GPU
    gpus = analyze_gpu()
    if gpus:
        actor_gpus = [g for g in gpus if g['mem_used'] > 50000]
        finetune_gpus = [g for g in gpus if 1000 < g['mem_used'] <= 50000]
        idle_gpus = [g for g in gpus if g['mem_used'] <= 1000]

        print(f"\n  GPU Status:")
        for g in gpus:
            role = "actor" if g['mem_used'] > 50000 else ("finetune" if g['mem_used'] > 1000 else "idle")
            bar = '█' * (g['util'] // 5) + '░' * (20 - g['util'] // 5)
            print(f"    GPU {g['idx']}: [{bar}] {g['util']:3d}%  {g['mem_used']:5d}/{g['mem_total']} MiB  ({role})")

    # Rollouts
    r = analyze_rollouts(run_dir)
    if r:
        print(f"\n  Rollout Generation:")
        print(f"    Started: {r.get('rollouts_started', '?')}, Finished: {r['rollouts_finished']}")
        print(f"    Elapsed: {r.get('elapsed_min', 0):.1f} min")
        rate = r.get('rollout_rate_per_min', 0)
        if rate > 0:
            print(f"    Rate: {rate:.1f} rollouts/min")
            print(f"    Avg time per rollout (est): {r.get('avg_rollout_time_min', 0):.1f} min")
        else:
            print(f"    Rate: waiting for more completions...")

    # Preprocess
    p = analyze_preprocess(run_dir)

    # Grader
    g = analyze_grader(run_dir)
    if g and g.get("total_grading_calls", 0) > 0:
        print(f"\n  Grader (Azure GPT-5.2):")
        print(f"    Total calls: {g['total_grading_calls']} ({g.get('errors', 0)} errors)")
        if g.get("score_distribution"):
            dist = g["score_distribution"]
            dist_str = ", ".join(f"s{k}={v}" for k, v in sorted(dist.items()))
            nz = g.get('nonzero_scored', 0)
            total = g.get('total_scored', 0)
            print(f"    Scores: {dist_str}  ({nz}/{total} non-zero = {nz/total*100:.0f}%)" if total > 0 else "")
        if g.get("avg_latency_sec"):
            print(f"    Latency: avg {g['avg_latency_sec']:.1f}s, median {g['median_latency_sec']:.1f}s, "
                  f"p90 {g['p90_latency_sec']:.1f}s, range [{g['min_latency_sec']:.1f}s, {g['max_latency_sec']:.1f}s]")
        if g.get("grading_rate_per_min"):
            print(f"    Throughput: {g['grading_rate_per_min']:.1f} gradings/min")
        if g.get("avg_response_bytes"):
            print(f"    Avg response size: {g['avg_response_bytes']:.0f} bytes")
        if g.get("rate_limit_usage_pct") is not None:
            print(f"    Rate limit usage: {g['rate_limit_usage_pct']:.1f}% "
                  f"({g.get('rate_limit_remaining_tokens', '?')}/{g.get('rate_limit_tokens', '?')} tokens remaining)")

    if p:
        print(f"\n  Preprocessor (graded samples → training data):")
        print(f"    Samples: {p['samples_ready']} / {p['target']}")
        if p.get('sample_rate_per_min'):
            print(f"    Rate: {p['sample_rate_per_min']:.1f} samples/min")
            if p.get('est_minutes_to_512') and p['est_minutes_to_512'] < 1e6:
                print(f"    ETA to 512: ~{p['est_minutes_to_512']:.0f} min")
        if p.get('avg_preprocess_sec_per_sample'):
            print(f"    Preprocess speed: {p['avg_preprocess_sec_per_sample']:.1f} sec/sample")

    # Finetune
    f = analyze_finetune(run_dir)
    if f:
        print(f"\n  Finetune:")
        if f.get('completed_steps'):
            print(f"    Completed steps: {f['completed_steps']}")
            print(f"    Latest: reward={f['latest_reward']:.3f}, loss={f['latest_loss']:.1f}, "
                  f"grad_norm={f.get('latest_grad_norm', 0):.3f}")
            print(f"    Sec/pass: {f['latest_sec_per_pass']:.1f}s, passes: {f.get('latest_passes', 0):.0f}")
            print(f"    Step time: {f['latest_sec_per_step']:.0f}s ({f['latest_sec_per_step']/3600:.1f}h), "
                  f"waiting: {f['latest_time_waiting']:.0f}s ({f['latest_time_waiting']/3600:.1f}h)")
            if f.get('reward_history'):
                print(f"    Reward history: {' -> '.join(f'{x:.3f}' for x in f['reward_history'])}")
            if f.get('all_steps'):
                loss_hist = [s['loss'] for s in f['all_steps']]
                norm_hist = [s['grad_norm'] for s in f['all_steps']]
                step_times = [s['sec_per_step'] for s in f['all_steps']]
                print(f"    Loss history:   {' -> '.join(f'{x:.1f}' for x in loss_hist)}")
                print(f"    Norm history:   {' -> '.join(f'{x:.3f}' for x in norm_hist)}")
                avg_step = sum(step_times) / len(step_times)
                print(f"    Avg step time:  {avg_step:.0f}s ({avg_step/3600:.2f}h)")
            if f.get('current_step'):
                print(f"    Current: step {f['current_step']} in progress ({f.get('current_passes', 0)} passes done)")
        else:
            status = f.get('status', 'unknown')
            print(f"    Status: {status}")

    # Summary
    print(f"\n  Pipeline Summary:")
    finished = r.get('rollouts_finished', 0)
    samples = p.get('samples_ready', 0)
    steps = f.get('completed_steps', 0)
    rate = r.get('rollout_rate_per_min', 0)
    sample_rate = p.get('sample_rate_per_min', 0)
    graded = g.get('total_grading_calls', 0)

    print(f"    Rollouts → Grader → Preprocess → Finetune")
    print(f"    {finished:>6}    {graded:>5}    {samples:>4}/512      Step {steps}")

    if sample_rate > 0 and samples < 512:
        eta = (512 - samples) / sample_rate
        print(f"    ETA to next finetune step: ~{eta:.0f} min ({eta/60:.1f}h)")
    elif graded > 0 and samples == 0:
        print(f"    Grading active ({graded} scored), waiting for groups to complete (need 16 rollouts/group)")
    elif rate > 0 and graded == 0:
        print(f"    Rollouts generating, grading not yet started")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest_run()
    if not run_dir:
        print("No run found")
        sys.exit(1)
    print_report(run_dir)
