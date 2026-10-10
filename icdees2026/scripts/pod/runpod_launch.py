"""Run the pod job lists on RunPod secure RTX 4090s, pull results back, terminate every pod.

One thread per pod: deploy -> wait for SSH -> upload -> start run_jobs.sh -> every 60 s pull
results/ and run.log -> on ALL_DONE / SETUP_FAILED / time cap: final pull, terminate.
The RunPod API token is read from the RUNPOD_API_TOKEN environment variable.
Usage: runpod_launch.py a b c      (letters select jobs_<x>.txt)
"""
import os, sys, json, time, subprocess, threading, urllib.request

W = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(W, "pod_results"); os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(W, "runpod_launch.log"), "a", buffering=1, encoding="utf-8")
GPU, IMAGE = "NVIDIA GeForce RTX 4090", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CAP_S = float(os.environ.get("CAP_H", "2.4")) * 3600          # per-pod uptime cap; every pod is terminated at this point
KEY = os.path.expanduser("~/.ssh/id_ed25519"); PUB = open(KEY + ".pub").read().strip()
UPLOAD = ["train_split.py", "run_jobs.sh", "splits.npz", "pod_corpus.csv.gz"]

def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(line, flush=True); LOG.write(line + "\n")

TOKEN = os.environ.get("RUNPOD_API_TOKEN", "")
assert TOKEN, "no RunPod token"

def gql(q):
    req = urllib.request.Request("https://api.runpod.io/graphql", data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN, "User-Agent": "curl/8.9"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    if d.get("errors"): raise RuntimeError(d["errors"])
    return d["data"]

def ssh_args(ip, port):
    return ["-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ConnectTimeout=15", "-o", "BatchMode=yes"] + (["-p", str(port)] if port else [])

def sh(ip, port, c, t=120, tries=3):
    for i in range(tries):                         # a busy host can stall one ssh call; retry before giving up
        try:
            return subprocess.run(["ssh", *ssh_args(ip, port), "root@" + ip, c], capture_output=True, text=True, timeout=t)
        except subprocess.TimeoutExpired:
            if i == tries - 1: raise
            time.sleep(10)

def pull(ip, port, x):
    d = os.path.join(RES, x); os.makedirs(d, exist_ok=True)
    subprocess.run(["scp", *ssh_args(ip, None), "-P", str(port), "-r", "root@%s:/workspace/job/results/." % ip, d], capture_output=True, timeout=600)
    for f in ("run.log", "setup.log", "cap_check.log"):
        subprocess.run(["scp", *ssh_args(ip, None), "-P", str(port), "root@%s:/workspace/job/%s" % (ip, f), d], capture_output=True, timeout=120)

def deploy(x):
    for attempt in range(20):                                       # stock can be momentarily short
        try:
            d = gql('mutation { podFindAndDeployOnDemand(input: {cloudType: SECURE, gpuCount: 1, volumeInGb: 0, '
                    'containerDiskInGb: 40, minVcpuCount: 6, minMemoryInGb: 24, gpuTypeId: "%s", name: "ihd-%s", '
                    'imageName: "%s", ports: "22/tcp", supportPublicIp: true, startSsh: true, '
                    'env: [{key: "PUBLIC_KEY", value: "%s"}]}) { id costPerHr } }' % (GPU, x, IMAGE, PUB))
            pod = d["podFindAndDeployOnDemand"]; log(x, "pod", pod["id"], "deployed at $%s/hr" % pod["costPerHr"]); return pod["id"]
        except Exception as e:
            log(x, "deploy retry:", str(e)[:160]); time.sleep(30)
    raise RuntimeError("could not deploy")

def endpoint(pid, x):
    t0 = time.time()
    while time.time() - t0 < 900:
        r = gql('query { pod(input: {podId: "%s"}) { runtime { ports { ip isIpPublic privatePort publicPort } } } }' % pid)["pod"]["runtime"]
        ports = [p for p in (r or {}).get("ports") or [] if p["privatePort"] == 22 and p["isIpPublic"]]
        if ports and sh(ports[0]["ip"], ports[0]["publicPort"], "echo ok", 30).stdout.strip() == "ok":
            log(x, "ssh ready", ports[0]["ip"], ports[0]["publicPort"]); return ports[0]["ip"], ports[0]["publicPort"]
        time.sleep(15)
    raise RuntimeError("ssh never came up")

def uptime(pid):
    r = gql('query { pod(input: {podId: "%s"}) { runtime { uptimeInSeconds } } }' % pid)["pod"]["runtime"]
    return (r or {}).get("uptimeInSeconds") or 0

def ensure_started(x, ip, port):
    """Upload whatever is missing or partial (compared by size), then start run_jobs.sh unless it already runs."""
    sh(ip, port, "mkdir -p /workspace/job < /dev/null")
    files = {f: os.path.join(W, f) for f in UPLOAD}; files["jobs.txt"] = os.path.join(W, "jobs_%s.txt" % x)
    for remote, local in files.items():
        have = sh(ip, port, "stat -c %%s /workspace/job/%s 2>/dev/null < /dev/null" % remote).stdout.strip()
        if have != str(os.path.getsize(local)):
            log(x, "uploading", remote)
            subprocess.run(["scp", *ssh_args(ip, None), "-P", str(port), local, "root@%s:/workspace/job/%s" % (ip, remote)], check=True, timeout=3600)
    if sh(ip, port, "ps -eo args | grep -q '^bash run_jobs[.]sh' && echo running < /dev/null").stdout.strip() != "running":
        try:      # the subshell form returns at once; a slow return is not a failure, ps below decides
            sh(ip, port, "cd /workspace/job && (setsid nohup bash run_jobs.sh > nohup.out 2>&1 < /dev/null &)", 30)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(5)
        ok = sh(ip, port, "ps -eo args | grep -q '^bash run_jobs[.]sh' && echo running < /dev/null").stdout.strip() == "running"
        if not ok: raise RuntimeError("run_jobs.sh did not start")
        log(x, "jobs started")
    else:
        log(x, "jobs already running")

def worker(x, pid=None, ip=None, port=None):
    try:
        pid = pid or deploy(x)
        if not ip: ip, port = endpoint(pid, x)
        ensure_started(x, ip, port)
        seen = set()
        while True:
            time.sleep(60)
            try:
                tail = sh(ip, port, "cat /workspace/job/run.log 2>/dev/null; tail -2 /workspace/job/setup.log 2>/dev/null < /dev/null").stdout
                up = uptime(pid)
            except Exception as e:
                log(x, "poll error", repr(e)[:120]); continue
            for line in tail.splitlines():
                if line.startswith(("###", "FAILED", "SETUP_FAILED", "READY", "Traceback")) and line not in seen:
                    seen.add(line); log(x, line[:200])
            if "###" in tail: pull(ip, port, x)
            if "ALL_DONE" in tail or "SETUP_FAILED" in tail or up > CAP_S:
                pull(ip, port, x); log(x, "finished" if "ALL_DONE" in tail else "STOPPED (setup failure or %.1f h cap)" % (CAP_S / 3600)); break
    except Exception as e:
        log(x, "ERROR", repr(e)[:300])
    finally:
        if pid:
            try: gql('mutation { podTerminate(input: {podId: "%s"}) }' % pid); log(x, "pod", pid, "terminated")
            except Exception as e: log(x, "TERMINATE FAILED", pid, e)

if __name__ == "__main__":
    # runpod_launch.py a b c                    -> deploy fresh pods
    # runpod_launch.py attach x:pod_id:ip:port  -> take over pods that are already running
    if sys.argv[1] == "attach":
        specs = [a.split(":") for a in sys.argv[2:]]
        ts = [threading.Thread(target=worker, args=(x, pid, ip, int(port))) for x, pid, ip, port in specs]
    else:
        ts = [threading.Thread(target=worker, args=(x,)) for x in sys.argv[1:]]
    for t in ts: t.start(); time.sleep(5)
    for t in ts: t.join()
    m = gql("query { myself { clientBalance currentSpendPerHr pods { id desiredStatus } } }")["myself"]
    log("ALL PODS DONE | balance $%.2f | spend/hr $%s | pods left %s" % (m["clientBalance"], m["currentSpendPerHr"], m["pods"]))
