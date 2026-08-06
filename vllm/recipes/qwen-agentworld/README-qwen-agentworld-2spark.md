# Qwen-AgentWorld-35B-A3B on 2× DGX Spark

Tensor-parallel inference across two Sparks (1× GB10 each). This matches the
model card's `--tensor-parallel-size 4` intent scaled to two GPUs: **TP=2**.

## Topology

| Node | `NODE_RANK` | Role | GPUs |
|------|-------------|------|------|
| Spark A (head) | 0 | API + engine rank 0 | 1 |
| Spark B (worker) | 1 | `--headless` engine rank 1 | 1 |

**World size** = `tensor_parallel_size` × `pipeline_parallel_size` = 2 × 1 = 2.

## Venv

Both nodes need the same stock vLLM venv. See **[readme-venv.md](./readme-venv.md)**.

## Prerequisites (both Sparks)

1. **Connect Two Sparks** — QSFP link, interconnect IPs, passwordless SSH:
   [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)
2. **Same vLLM venv** on both nodes — [readme-venv.md](./readme-venv.md)
3. **Hugging Face cache** — download once or replicate:
   ```bash
   huggingface-cli download Qwen/Qwen-AgentWorld-35B-A3B
   ```
4. **Firewall** — allow TCP on interconnect (`MASTER_PORT` default 29501, NCCL ephemeral ports).
5. **Clock sync** (NTP) on both nodes.

## Network

Follow [NCCL for Two Sparks — Run on two Sparks](https://build.nvidia.com/spark/nccl/stacked-sparks):

```bash
ip -br link    # use an enp1* interface that is Up (not enP2p* alias)
ip -br addr    # interconnect IPs on that interface
```

The launch script exports the same variables as the NCCL test playbook, plus `GLOO_SOCKET_IFNAME` (fixes Gloo binding to `127.0.0.1`):

| Variable | Purpose |
|----------|---------|
| `HOST_IP` / `VLLM_HOST_IP` | This node's interconnect IP |
| `HEAD_ADDR` / `MASTER_ADDR` | Head interconnect IP |
| `NCCL_IFNAME` | QSFP interface (e.g. `enp1s0f0np0`) |
| `NCCL_SOCKET_IFNAME` | NCCL traffic |
| `GLOO_SOCKET_IFNAME` | PyTorch Gloo store (CPU process group) |
| `UCX_NET_DEVICES` | UCX / MPI transport |
| `OMPI_MCA_btl_tcp_if_include` | Open MPI TCP binding |

In `qwen-agentworld-2spark.env`:

- `HEAD_ADDR` — head interconnect IP
- `HOST_IP` — this machine's interconnect IP
- `NCCL_IFNAME` — QSFP interface name (Up in `ip -br link`)

## First-time boot notes

From single-Spark runs:

- Use **`--language-model-only`** (checkpoint has no vision weights).
- First launch **JIT-compiles FlashInfer MoE kernels** (15–30+ min). Do not interrupt.
- **`earlyoom`** may kill vLLM when RAM hits 2% during compile — on **both** nodes either:
  - set `STOP_EARLYOOM=1` in the env file for first boot, or
  - `sudo systemctl stop earlyoom` until the server is up.
- `MAX_JOBS=4` limits parallel FlashInfer `nvcc` jobs (ninja `-j N`)
- `FLASHINFER_NVCC_THREADS=2` sets `nvcc --threads=N` per job
- `OMP_NUM_THREADS=4` avoids vLLM forcing torch to 1 thread during compile

## Launch

On **both** Sparks:

```bash
cd vllm/recipes/qwen-agentworld
cp qwen-agentworld-2spark.env.example qwen-agentworld-2spark.env
# edit HEAD_ADDR, HOST_IP, NODE_RANK, VLLM_VENV
```

**Spark A (head, NODE_RANK=0):**

```bash
./qwen-agentworld-2spark.sh head
```

**Spark B (worker, NODE_RANK=1)** — after head begins initializing:

```bash
./qwen-agentworld-2spark.sh worker
```

Wait until head logs `Application startup complete` or Uvicorn listening on `:8000`.

## Verify

From any machine that can reach the head:

```bash
curl -s "http://${HEAD_ADDR}:8000/v1/models" | jq .
```

Chat completion:

```bash
curl -s "http://${HEAD_ADDR}:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-AgentWorld-35B-A3B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

## Context length

| Setting | Notes |
|---------|--------|
| `MAX_MODEL_LEN=131072` | Good starting point with TP=2 (~32 GiB weights/GPU) |
| `262144` | Model default; try after stable boot if KV fits |
| `62144` | Conservative if OOM during profiling |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `World size (2) > available GPUs (1)` | Missing `--nnodes 2` / worker not running |
| Worker cannot connect | Start **head first**; check interconnect ping; verify `GLOO_SOCKET_IFNAME` / `NCCL_IFNAME` |
| Gloo `127.0.0.1` connection refused | Set `GLOO_SOCKET_IFNAME` to QSFP iface (script does this); wrong `HOST_IP` |
| vLLM version mismatch | Match `EXPECTED_VLLM_VERSION` on both nodes (0.23.0) |
| `earlyoom` SIGTERM to `VLLM::EngineCor` | Stop earlyoom or reduce `MAX_JOBS` during first compile |
| Vision weight errors | Add `--language-model-only` |
| NCCL hang | Set `NCCL_DEBUG=INFO`, verify `NCCL_IFNAME`, ping between IPs |

## Manual commands (equivalent)

**Head:**

```bash
export VLLM_HOST_IP=<HOST_IP> MASTER_ADDR=<HEAD_IP> MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=<IFACE> GLOO_SOCKET_IFNAME=<IFACE>
export UCX_NET_DEVICES=<IFACE> OMPI_MCA_btl_tcp_if_include=<IFACE>
export MAX_JOBS=2
source ~/vllm/.venv/bin/activate

vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 2 \
  --nnodes 2 --node-rank 0 \
  --master-addr <HEAD_IP> --master-port 29501 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --language-model-only \
  --distributed-executor-backend mp
```

**Worker:**

```bash
export VLLM_HOST_IP=<WORKER_IP> MASTER_ADDR=<HEAD_IP> MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=<IFACE> GLOO_SOCKET_IFNAME=<IFACE>
export UCX_NET_DEVICES=<IFACE> OMPI_MCA_btl_tcp_if_include=<IFACE>
export MAX_JOBS=2
source ~/vllm/.venv/bin/activate

vllm serve Qwen/Qwen-AgentWorld-35B-A3B \
  --tensor-parallel-size 2 \
  --nnodes 2 --node-rank 1 \
  --master-addr <HEAD_IP> --master-port 29501 \
  --max-model-len 131072 \
  --reasoning-parser qwen3 \
  --trust-remote-code \
  --language-model-only \
  --distributed-executor-backend mp \
  --headless
```
