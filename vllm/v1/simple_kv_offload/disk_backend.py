# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.
"""

from __future__ import annotations

import contextlib
import os
import queue
import threading

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload import profiler
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096
# Default coalesced-I/O group size: one full pwritev/preadv per contiguous run
# chunk of up to this many bytes (16 MiB @ 64 KiB blocks = 256 slots).
_DEFAULT_SEGMENT_BYTES = 16 * (1024**2)
# os.pwritev/preadv accept at most IOV_MAX (1024) iov entries per call.
_IOV_MAX = 1024


def _find_runs(blocks: list[int]) -> list[tuple[int, int]]:
    """把块 id 序列切成连续 run：返回 [(起始索引, 长度), ...]。

    run 内块 id 严格 +1 递增（磁盘 slot 连续 => 文件偏移连续），可合并为
    单次大 I/O；run 间边界即 I/O 切分点。对任意布局正确：碎片布局退化为
    多个短 run，仍不劣于逐块路径。
    """
    if not blocks:
        return []
    runs: list[tuple[int, int]] = []
    run_start = 0
    for i in range(1, len(blocks)):
        if blocks[i] != blocks[i - 1] + 1:
            runs.append((run_start, i - run_start))
            run_start = i
    runs.append((run_start, len(blocks) - run_start))
    return runs


def _alloc_aligned(num_slots: int, bpb: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * bpb
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, bpb)


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        # KVLog M3 阶段一：run 合并 I/O。>0 表示合并路径的每半缓冲 slot 数。
        self._coalesce_half: int = 0

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
        coalesce_io: bool = True,
        segment_bytes: int = _DEFAULT_SEGMENT_BYTES,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._tensor_names = list(gpu_caches.keys())
        if coalesce_io:
            # 组缓冲（段级双缓冲）：每半缓冲容纳 segment_bytes 的连续 slot，
            # 受 IOV_MAX/每 slot iov 条目数上限约束。
            seg_slots = max(1, -(-segment_bytes // total_block_bytes))
            iov_cap = max(1, _IOV_MAX // max(1, len(gpu_caches)))
            self._coalesce_half = min(seg_slots, iov_cap)
            num_buffer_slots = max(num_buffer_slots, 2 * self._coalesce_half)
        self._num_buffer_slots = num_buffer_slots
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        assert total_block_bytes % _ALIGNMENT == 0, (
            f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
        )

        # Separate buffer pools for store and load threads
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            store_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(store_buf)
            self._store_buffer_caches[name] = store_buf
            load_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(load_buf)
            self._load_buffer_caches[name] = load_buf

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [
                memoryview(self._store_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [
                memoryview(self._load_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]

        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_disk_slots * total_block_bytes)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s, coalesce_half=%d)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
            use_page_cache,
            self._coalesce_half,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        if self._fd < 0:
            return
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        os.close(self._fd)
        self._fd = -1

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            try:
                if wait_event is not None:
                    stream.wait_event(wait_event)
                self._do_store(src_blocks, dst_blocks, stream)
            except Exception:
                # 后台线程死亡必须可见，否则表现为 store 计数静默归零
                logger.exception(
                    "DiskBackend store thread exiting: %d blocks lost",
                    len(src_blocks),
                )
                return
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        written = os.pwritev(self._fd, self._store_slot_views[buf_slot], file_offset)
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("store", written)

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        bytes_read = os.preadv(self._fd, self._load_slot_views[buf_slot], file_offset)
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("load", bytes_read)

    def _writev_range(self, buf_start: int, k: int, file_offset: int) -> None:
        """一次 pwritev 写出连续 k 个缓冲 slot（对应磁盘上的一段连续 run）。"""
        views: list[memoryview] = []
        for slot in range(buf_start, buf_start + k):
            views.extend(self._store_slot_views[slot])
        nbytes = k * self._total_block_bytes
        written = os.pwritev(self._fd, views, file_offset)
        if written < nbytes:
            raise OSError(
                f"Short write: expected {nbytes} bytes, wrote {written}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("store", written)

    def _readv_range(self, buf_start: int, k: int, file_offset: int) -> None:
        """一次 preadv 读入连续 k 个缓冲 slot（磁盘上一段连续 run）。"""
        views: list[memoryview] = []
        for slot in range(buf_start, buf_start + k):
            views.extend(self._load_slot_views[slot])
        nbytes = k * self._total_block_bytes
        bytes_read = os.preadv(self._fd, views, file_offset)
        if bytes_read < nbytes:
            raise OSError(
                f"Short read: expected {nbytes} bytes, read {bytes_read}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("load", bytes_read)

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            try:
                if wait_event is not None:
                    stream.wait_event(wait_event)
                self._do_load(src_blocks, dst_blocks, stream)
            except Exception:
                # 后台线程死亡必须可见，否则表现为 load 计数静默归零
                logger.exception(
                    "DiskBackend load thread exiting: %d blocks lost",
                    len(src_blocks),
                )
                return
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        if self._coalesce_half > 0:
            self._do_store_coalesced(gpu_blocks, disk_slots, stream)
            return
        self._do_store_per_block(gpu_blocks, disk_slots, stream)

    def _do_store_coalesced(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """run 合并写：每 run chunk 一次批量 DMA + 一次大 pwritev。

        段级双缓冲（两半组缓冲轮换）：先发本 chunk DMA，再等上一 chunk 的
        DMA 事件并落盘，使 DMA 与磁盘 I/O 流水重叠。块粒度上 syscall 数
        从 N 次 pwritev 降为 run-chunk 数。
        """
        assert self._store_params is not None
        half = self._coalesce_half
        # KVLog profiling 计时器（KVLOG_PROFILE=0 时为静态零开销）
        prof = profiler.PROFILE
        t0 = profiler.now() if prof else 0.0
        io_t = sync_t = dma_t = 0.0

        # (gpu 起始索引, 磁盘起始 slot, 块数)：run × chunk 展开
        ops: list[tuple[int, int, int]] = []
        runs = _find_runs(disk_slots)
        for start, length in runs:
            for off in range(0, length, half):
                k = min(half, length - off)
                ops.append((start + off, disk_slots[start + off], k))
        if prof:
            # run/chunk 长度分布：诊断 syscall 粒度（写路径打断 vs 缓冲上限）
            profiler.note_runs(
                "store", [length for _, length in runs],
                [k for _, _, k in ops],
            )

        # (DMA event, buf_start, file_offset, k)：已 DMA 完成待落盘的 chunk
        pending: tuple[torch.Event, int, int, int] | None = None
        half_idx = 0
        for idx0, disk_start, k in ops:
            buf_start = half_idx * half
            half_idx ^= 1
            # 先发本 chunk DMA（进连续 buffer 区），再刷上一 chunk —— 重叠
            if prof:
                ts = profiler.now()
            copy_blocks(
                gpu_blocks[idx0 : idx0 + k],
                list(range(buf_start, buf_start + k)),
                self._store_params,
                coalesce=True,
            )
            if prof:
                dma_t += profiler.now() - ts
            ev = torch.Event()
            ev.record(stream)
            if pending is not None:
                pev, pbs, poff, pk = pending
                if prof:
                    ts = profiler.now()
                pev.synchronize()
                if prof:
                    sync_t += profiler.now() - ts
                    ts = profiler.now()
                self._writev_range(pbs, pk, poff)
                if prof:
                    io_t += profiler.now() - ts
            pending = (ev, buf_start, disk_start * self._total_block_bytes, k)

        if pending is not None:
            pev, pbs, poff, pk = pending
            if prof:
                ts = profiler.now()
            pev.synchronize()
            if prof:
                sync_t += profiler.now() - ts
                ts = profiler.now()
            self._writev_range(pbs, pk, poff)
            if prof:
                io_t += profiler.now() - ts

        if prof:
            profiler.note_batch(
                "store", len(gpu_blocks),
                profiler.now() - t0, io_t, sync_t, dma_t,
            )

    def _do_store_per_block(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """原生逐块路径（disk_coalesce_io=false 的 A/B 基线）。"""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, file offset) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n
        # KVLog profiling 计时器（KVLOG_PROFILE=0 时为静态零开销）
        prof = profiler.PROFILE
        t0 = profiler.now() if prof else 0.0
        io_t = sync_t = dma_t = 0.0

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                if prof:
                    ts = profiler.now()
                prev[0].synchronize()
                if prof:
                    sync_t += profiler.now() - ts
                    ts = profiler.now()
                self._writev_slot(buf_slot, prev[1])
                if prof:
                    io_t += profiler.now() - ts

            if prof:
                ts = profiler.now()
            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            if prof:
                dma_t += profiler.now() - ts
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot * self._total_block_bytes)

        for slot, last in enumerate(pending):
            if last is not None:
                if prof:
                    ts = profiler.now()
                last[0].synchronize()
                if prof:
                    sync_t += profiler.now() - ts
                    ts = profiler.now()
                self._writev_slot(slot, last[1])
                if prof:
                    io_t += profiler.now() - ts

        if prof:
            profiler.note_batch(
                "store", len(gpu_blocks),
                profiler.now() - t0, io_t, sync_t, dma_t,
            )

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        if self._coalesce_half > 0:
            self._do_load_coalesced(disk_slots, gpu_blocks, stream)
            return
        self._do_load_per_block(disk_slots, gpu_blocks, stream)

    def _do_load_coalesced(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """run 合并读：每 run chunk 一次大 preadv + 一次批量 DMA 回 GPU 散块。

        段级双缓冲：先等上一 chunk 的 DMA 事件（释放其缓冲半区），再读入
        本半区并发 DMA，readv 与上一 chunk 的 DMA 重叠。
        """
        assert self._load_params is not None
        half = self._coalesce_half
        # KVLog profiling 计时器（KVLOG_PROFILE=0 时为静态零开销）
        prof = profiler.PROFILE
        t0 = profiler.now() if prof else 0.0
        io_t = sync_t = dma_t = 0.0

        # (gpu 起始索引, 磁盘起始 slot, 块数)：run × chunk 展开
        ops: list[tuple[int, int, int]] = []
        runs = _find_runs(disk_slots)
        for start, length in runs:
            for off in range(0, length, half):
                k = min(half, length - off)
                ops.append((start + off, disk_slots[start + off], k))
        if prof:
            profiler.note_runs(
                "load", [length for _, length in runs],
                [k for _, _, k in ops],
            )

        pending: torch.Event | None = None  # 上一 chunk 的 DMA 事件
        half_idx = 0
        for idx0, disk_start, k in ops:
            buf_start = half_idx * half
            half_idx ^= 1
            if pending is not None:
                if prof:
                    ts = profiler.now()
                pending.synchronize()
                if prof:
                    sync_t += profiler.now() - ts

            if prof:
                ts = profiler.now()
            self._readv_range(buf_start, k, disk_start * self._total_block_bytes)
            if prof:
                io_t += profiler.now() - ts

            if prof:
                ts = profiler.now()
            copy_blocks(
                list(range(buf_start, buf_start + k)),
                gpu_blocks[idx0 : idx0 + k],
                self._load_params,
                coalesce=True,
            )
            if prof:
                dma_t += profiler.now() - ts
            ev = torch.Event()
            ev.record(stream)
            pending = ev

        if prof:
            profiler.note_batch(
                "load", len(disk_slots),
                profiler.now() - t0, io_t, sync_t, dma_t,
            )

    def _do_load_per_block(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """原生逐块路径（disk_coalesce_io=false 的 A/B 基线）。"""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n
        # KVLog profiling 计时器（KVLOG_PROFILE=0 时为静态零开销）
        prof = profiler.PROFILE
        t0 = profiler.now() if prof else 0.0
        io_t = sync_t = dma_t = 0.0

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                if prof:
                    ts = profiler.now()
                prev.synchronize()
                if prof:
                    sync_t += profiler.now() - ts

            if prof:
                ts = profiler.now()
            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)
            if prof:
                io_t += profiler.now() - ts

            if prof:
                ts = profiler.now()
            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            if prof:
                dma_t += profiler.now() - ts
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev

        if prof:
            profiler.note_batch(
                "load", len(disk_slots),
                profiler.now() - t0, io_t, sync_t, dma_t,
            )
