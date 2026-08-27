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


def now() -> float:
    return time.perf_counter()


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


def _dump() -> None:
    try:
        with _lock:
            payload = {
                "profile": {
                    "pid": os.getpid(),
                    "dump_ts": time.time(),
                    "io": {k: dict(v) for k, v in _io_stats.items()},
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
                    "load_events": _load_events[:_MAX_SERIES],
                    "pending_series": _pending_series[:_MAX_SERIES],
                    "flush": dict(_flush_stats),
                },
            }
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        tmp = OUT_PATH + ".tmp"
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
