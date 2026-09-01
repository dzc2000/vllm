# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVLog 只读 profiling 埋点（M3 步骤 0：Figure 1 生产线）。

激活方式：环境变量 KVLOG_PROFILE=1。
输出路径：KVLOG_PROFILE_OUT（默认 /tmp/kvlog_fig1_stats.json），进程退出时 atexit dump。
默认关闭时所有调用都是空操作或廉价 bool 检查，不改变任何行为。

记录内容：
- store/load 线程：每批 wall time、块数；io_time（pwritev/preadv 墙钟）、
  sync_time（CUDA event synchronize 墙钟）、dma_time（copy_blocks 提交墙钟）；
  syscall 计数与字节数。
- worker：load 提交->完成的 per-event 延迟、每步 poll 时的 pending load 数
  时间序列（"有多少请求在等 KV"曲线）、抢占 flush 阻塞时长。
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time

from vllm.logger import init_logger

PROFILE = os.environ.get("KVLOG_PROFILE", "0") == "1"
_BASE_PATH = os.environ.get("KVLOG_PROFILE_OUT", "/tmp/kvlog_fig1_stats.json")
OUT_PATH = _BASE_PATH

logger = init_logger(__name__)

_lock = threading.Lock()


def _new_direction() -> dict[str, float]:
    return {
        "batches": 0,
        "blocks": 0,
        "batch_wall_s": 0.0,
        "io_wall_s": 0.0,  # pwritev/preadv 墙钟累计
        "sync_wall_s": 0.0,  # event.synchronize() 墙钟累计
        "dma_wall_s": 0.0,  # copy_blocks 提交墙钟累计
        "syscalls": 0,
        "bytes": 0,
    }


# 磁盘后端计数（store / load 两个方向）
_io_stats: dict[str, dict[str, float]] = {
    "store": _new_direction(),
    "load": _new_direction(),
}


# run 长度直方图（仅合并路径采集）：解释 syscall 粒度来源。
# run = 磁盘 slot 连续段（分配器决定的原始可合并性）；
# chunk = 实际 syscall 单元（run 按 half 上限切块后）。
# 两者分布分开记：run 短 => 段式分配被打断（写路径问题）；
# run 长但 chunk 短 => 组缓冲上限截断（可调 disk_segment_bytes）。
_run_stats: dict[str, dict] = {
    d: {
        "runs_hist": {},  # 桶(bit_length) -> run 个数
        "chunks_hist": {},  # 桶 -> chunk（=syscall）个数
        "runs": 0,
        "run_blocks": 0,
        "chunks": 0,
        "chunk_blocks": 0,
    }
    for d in ("store", "load")
}


def _bucket(length: int) -> int:
    # 1->1, 2-3->2, 4-7->3, 8-15->4 ... 指数桶，键稳定可 JSON 化
    return max(1, int(length)).bit_length()


# worker 侧时间序列（截断保护，避免长跑撑爆内存）
_MAX_SERIES = 200_000
_load_events: list[list[float]] = []  # [submit_ts, done_ts]
_pending_series: list[list[float]] = []  # [ts, pending_load_events]
_flush_stats = {"count": 0, "wall_s": 0.0}

# 等待上游依赖事件（如 compute_done）的主机侧墙钟累计：把 sync_wall 拆成
# 依赖等待（调度节拍税）与 DMA 本体等待，供论文 store 侧分解引用。
_dep_wait: dict[str, float] = {"store": 0.0, "load": 0.0}

# 体积账（W:R 写放大与死写率的原始计数）：
# stored_blocks / dropped_blocks：写侧逐块决策计数（WriteGate 接入前
# dropped 恒为 0，接入后即闸门拒绝数）。
# _block_hits：块内容哈希 -> 该块被写盘后读回（load）的次数。
# 键域 <= 盘池容量（每哈希最多一条），长跑安全；读回侧聚合为直方图输出。
_volume = {"stored_blocks": 0, "dropped_blocks": 0}
_block_hits: dict[bytes, int] = {}

# WriteGate v3 闭环回路：控制器把在线死写率与档位轨迹写回这里，随分片
# 一起落盘（gate 决策发生在 EngineCore 进程，与 _block_hits 同进程同账本）。
_gate: dict = {}


def now() -> float:
    return time.perf_counter()


def note_dep_wait(direction: str, seconds: float) -> None:
    """等待上游依赖事件（如 compute_done）的主机侧耗时。"""
    if not PROFILE:
        return
    with _lock:
        _dep_wait[direction] += seconds


def note_store_decision(n_stored: int, n_dropped: int,
                        stored_hashes: list[bytes] | None = None) -> None:
    """写侧逐块决策记账：写盘 / 丢弃（WriteGate 拒绝）块数。

    stored_hashes 为实际写盘块的内容哈希，用于与读回侧对账
    （哪块写出去之后从未被读回 = 死写）。
    """
    if not PROFILE:
        return
    with _lock:
        _volume["stored_blocks"] += n_stored
        _volume["dropped_blocks"] += n_dropped
        if stored_hashes:
            for h in stored_hashes:
                _block_hits.setdefault(h, 0)


def note_block_loads(hashes: list[bytes]) -> None:
    """读回侧记账：每块内容哈希的命中（load）次数 +1。"""
    if not PROFILE:
        return
    with _lock:
        for h in hashes:
            _block_hits[h] = _block_hits.get(h, 0) + 1


def hit_counts(hashes: list[bytes]) -> list[int]:
    """批量查读回账本：每个哈希被 load 的次数（无记录 = 0）。

    单次取锁，供 WriteGate 在逐块扫描中按祖先链批量判定"该前缀家族
    历史上是否被复用"（账本判据 = 命中次数 > 0，与体积账同源）。
    注意：PROFILE 关闭时账本不记录，返回值恒为全 0，调用方不可把
    "无数据"当"无命中"（gate 侧已做显式校验）。
    """
    if not PROFILE or not hashes:
        return [0] * len(hashes)
    with _lock:
        return [_block_hits.get(h, 0) for h in hashes]


def was_ever_hit(h: bytes) -> bool:
    """块写盘后是否至少被读回过一次（False = 尚无命中记录/未记账）。"""
    if not PROFILE:
        return False
    with _lock:
        return _block_hits.get(h, 0) > 0


def note_gate_state(state: dict) -> None:
    """WriteGate 闭环控制器快照（覆盖写，随分片 dump）。"""
    if not PROFILE:
        return
    with _lock:
        _gate.clear()
        _gate.update(state)


def note_batch(direction: str, blocks: int, wall: float,
               io: float, sync: float, dma: float) -> None:
    if not PROFILE:
        return
    with _lock:
        s = _io_stats[direction]
        s["batches"] += 1
        s["blocks"] += blocks
        s["batch_wall_s"] += wall
        s["io_wall_s"] += io
        s["sync_wall_s"] += sync
        s["dma_wall_s"] += dma


def note_syscalls(direction: str, nbytes: int) -> None:
    if not PROFILE:
        return
    with _lock:
        s = _io_stats[direction]
        s["syscalls"] += 1
        s["bytes"] += nbytes


def note_runs(direction: str, run_lengths: list[int],
              chunk_lengths: list[int]) -> None:
    """记录一批传输的 run/chunk 长度分布（合并路径专用）。"""
    if not PROFILE:
        return
    with _lock:
        rs = _run_stats[direction]
        for length in run_lengths:
            b = _bucket(length)
            rs["runs_hist"][b] = rs["runs_hist"].get(b, 0) + 1
            rs["runs"] += 1
            rs["run_blocks"] += length
        for length in chunk_lengths:
            b = _bucket(length)
            rs["chunks_hist"][b] = rs["chunks_hist"].get(b, 0) + 1
            rs["chunks"] += 1
            rs["chunk_blocks"] += length


def note_load_event(submit_ts: float, done_ts: float) -> None:
    if not PROFILE:
        return
    with _lock:
        if len(_load_events) < _MAX_SERIES:
            _load_events.append([submit_ts, done_ts])


def note_pending(n_pending: int) -> None:
    if not PROFILE:
        return
    with _lock:
        if len(_pending_series) < _MAX_SERIES:
            _pending_series.append([now(), n_pending])


def note_flush(wall: float) -> None:
    if not PROFILE:
        return
    with _lock:
        _flush_stats["count"] += 1
        _flush_stats["wall_s"] += wall


_activated_pid: int | None = None


def activate(out_path: str | None = None) -> None:
    """显式激活 profiling（同一进程内幂等，fork 出的子进程会重新激活）。

    环境变量在 EngineCore 子进程中不可靠（spawn 时可能未继承）；
    更关键的是 EngineCore 经 fork 从 main 继承 _activated 状态与
    dumper 线程（线程不随 fork 复制），因此守卫必须按 pid 判断：
    子进程 pid 不同 -> 重新绑定 OUT_PATH 并重启 dumper。
    可靠激活路径：connector.__init__ 从 kv_connector_extra_config 读开关
    后调用本函数。
    """
    global PROFILE, OUT_PATH, _BASE_PATH, _activated_pid
    pid = os.getpid()
    logger.info(
        "KVLog profiler: activate() called pid=%s activated_pid=%s out_arg=%r",
        pid, _activated_pid, out_path,
    )
    if _activated_pid == pid:
        return
    PROFILE = True
    if out_path:
        _BASE_PATH = out_path
    OUT_PATH = f"{_BASE_PATH}.{pid}"
    logger.info("KVLog profiler: active pid=%s out=%s", pid, OUT_PATH)
    _dump()  # 立即落盘一次：分片出现即证明本进程已激活
    logger.info(
        "KVLog profiler: initial dump done pid=%s exists=%s",
        pid, os.path.exists(OUT_PATH),
    )
    atexit.register(_dump)
    _t = threading.Thread(target=_dumper_loop, daemon=True)
    _t.start()
    _activated_pid = pid


def _volume_summary() -> dict:
    # 块级哈希表不出原始哈希（不可 JSON 化且体积大），聚合为：
    # 命中次数直方图（0 次 = 死写）+ 死写块数 + W:R 所需计数。
    hits_hist: dict[int, int] = {}
    dead = 0
    for n in _block_hits.values():
        hits_hist[n] = hits_hist.get(n, 0) + 1
        if n == 0:
            dead += 1
    return {
        "stored_blocks": _volume["stored_blocks"],
        "dropped_blocks": _volume["dropped_blocks"],
        "tracked_blocks": len(_block_hits),
        "dead_blocks": dead,
        "hits_hist": {str(k): v for k, v in sorted(hits_hist.items())},
    }


def _dump() -> None:
    try:
        with _lock:
            payload = {
                "profile": {
                    "pid": os.getpid(),
                    "dump_ts": time.time(),
                    "io": {k: dict(v) for k, v in _io_stats.items()},
                    "volume": _volume_summary(),
                    "gate": dict(_gate),
                    "runs": {
                        k: {
                            "runs_hist": dict(v["runs_hist"]),
                            "chunks_hist": dict(v["chunks_hist"]),
                            "runs": v["runs"],
                            "run_blocks": v["run_blocks"],
                            "chunks": v["chunks"],
                            "chunk_blocks": v["chunk_blocks"],
                        }
                        for k, v in _run_stats.items()
                    },
                    "dep_wait": dict(_dep_wait),
                    "load_events": _load_events[:_MAX_SERIES],
                    "pending_series": _pending_series[:_MAX_SERIES],
                    "flush": dict(_flush_stats),
                },
            }
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        # 每个写者独立 tmp：dumper 线程与 atexit 并发 _dump 时共用同一
        # tmp 会交错写入产生损坏 JSON（读方 json.loads 失败被跳过）。
        tmp = OUT_PATH + f".tmp.{threading.get_ident()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, OUT_PATH)
    except Exception:  # noqa: BLE001 - profiling 永不阻断主流程
        # 静默吞错曾导致分片缺失被误诊为"未激活"——必须可见
        logger.exception("KVLog profiler: dump failed pid=%s out=%s",
                         os.getpid(), OUT_PATH)


def _dumper_loop() -> None:
    # 常驻落盘：EngineCore 被 SIGTERM/os._exit 关闭时 atexit 不可靠，
    # 必须由后台线程持续刷新文件，读方才能拿到实时计数。
    while True:
        time.sleep(1.0)
        _dump()


if PROFILE:
    # main 进程与 EngineCore 子进程都会 import 本模块（LLM 构造时经
    # factory -> connector -> worker 的 import 链）。两进程共写一个文件会
    # 互相覆盖（main 恒为零值），因此每进程写独立 pid 文件，读取方合并。
    # 注意：spawn 的子进程可能未继承环境变量，EngineCore 侧的可靠激活
    # 路径是 connector.__init__ -> activate()。
    activate()
