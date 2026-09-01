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


_SEGMENT_SIZE = 32  # 与 manager._DiskSegmentAllocator 保持一致


def _reorder_by_segment(
    gpu_blocks: list[int], disk_slots: list[int]
) -> tuple[list[int], list[int], list[int]]:
    """阶段二：store 前按磁盘段分组重排 (gpu, slot) 对。

    多请求交错提交时 slot 序列跨段跳动（§3.7：74.6% 单块 run），但每对
    (gpu, slot) 落盘互不依赖——重排仅改变 DMA/写盘顺序，不改数据归属。
    按 ``slot // 32`` 聚类 + 段内排序后，同段 slot 变成连续 run，让
    pwritev 合并成段粒度大 I/O。

    返回 (重排后 gpu_blocks, 重排后 disk_slots, 旧索引->新索引映射)。
    映射保留供调用方对齐 store 事件回调里的块序（当前实现按对整体完成，
    无逐块回调，返回仅为可观测性）。
    """
    order = sorted(range(len(disk_slots)), key=lambda i: disk_slots[i])
    new_gpu = [gpu_blocks[i] for i in order]
    new_slots = [disk_slots[i] for i in order]
    return new_gpu, new_slots, order


def _alloc_aligned_flat(nbytes: int) -> torch.Tensor:
    """Allocate a flat staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes]


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
        self._store_slot_views: list[memoryview] = []
        self._load_slot_views: list[memoryview] = []
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
            # 组缓冲（段级双缓冲）：每半缓冲容纳 segment_bytes 的连续 slot。
            # 阶段 1.5：交错暂存缓冲使每个 chunk 只需 1 条 iov，
            # IOV_MAX 不再约束 chunk 上限，segment_bytes 可超 16 MiB。
            seg_slots = max(1, -(-segment_bytes // total_block_bytes))
            self._coalesce_half = seg_slots
            num_buffer_slots = max(num_buffer_slots, 2 * self._coalesce_half)
        self._num_buffer_slots = num_buffer_slots
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        assert total_block_bytes % _ALIGNMENT == 0, (
            f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
        )

        # 阶段 1.5 交错暂存缓冲：store/load 各一块连续内存，slot 内张量
        # 布局与磁盘一致（t0b0..tNb0, t0b1..tNb1, ...），因此任意连续 k 个
        # slot 在内存与磁盘上都是连续区段 -> 1 条 iov、1 次 syscall。
        # DMA 侧用 as_strided 按张量切视图（stride=total_block_bytes），
        # copy_blocks 按双侧 stride 寻址、逐块描述符搬运。
        assert sum(self._per_tensor_bpb) == total_block_bytes, (
            "total_block_bytes must equal the sum of per-tensor block bytes"
        )
        buf_bytes = num_buffer_slots * total_block_bytes
        store_flat = _alloc_aligned_flat(buf_bytes)
        load_flat = _alloc_aligned_flat(buf_bytes)
        pin_tensor(store_flat)
        pin_tensor(load_flat)
        self._store_flat = store_flat
        self._load_flat = load_flat
        self._store_np = store_flat.numpy()
        self._load_np = load_flat.numpy()

        # as_strided 的 storage_offset 是底层存储的绝对偏移，需加上对齐
        # 视图自身的偏移。
        store_off = store_flat.storage_offset()
        load_off = load_flat.storage_offset()
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        off = 0
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            self._store_buffer_caches[name] = torch.as_strided(
                store_flat, (num_buffer_slots, bpb),
                (total_block_bytes, 1), store_off + off,
            )
            self._load_buffer_caches[name] = torch.as_strided(
                load_flat, (num_buffer_slots, bpb),
                (total_block_bytes, 1), load_off + off,
            )
            off += bpb

        # 每 slot 单条 iov（slot 在交错缓冲中连续）
        ttb = total_block_bytes
        self._store_slot_views = [
            memoryview(self._store_np[s * ttb : (s + 1) * ttb])
            for s in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            memoryview(self._load_np[s * ttb : (s + 1) * ttb])
            for s in range(num_buffer_slots)
        ]

        # KVLog 探针：KVLOG_STORE_SRC_ANY=1 时 store DMA 源访问顺序改 ANY，
        # 验证 STREAM 语义被前向 kernel 队列阻塞的假说。注意 ANY 仅对
        # 拷贝期间不再变化的源是安全的（此处探针用途，默认仍 STREAM）。
        store_src_order = (
            CU_MEMCPY_SRC_ACCESS_ORDER_ANY
            if os.environ.get("KVLOG_STORE_SRC_ANY", "0") == "1"
            else CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        )
        logger.info(
            "KVLog store DMA src_access_order=%s (KVLOG_STORE_SRC_ANY=%s)",
            "ANY" if store_src_order == CU_MEMCPY_SRC_ACCESS_ORDER_ANY else "STREAM",
            os.environ.get("KVLOG_STORE_SRC_ANY", "0"),
        )
        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=store_src_order,
            copy_sizes=self._per_tensor_bpb,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
            copy_sizes=self._per_tensor_bpb,
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

    def store_barrier(self, timeout: float = 30.0) -> bool:
        """等 store 线程处理完此前入队的所有任务（含 done 事件注册）。

        队列 FIFO：屏障项排在先入队的批之后，被 store 线程取到并
        set 时，先期各批的 done 事件必已 append 进 events_list。
        供抢占 flush 在同步事件前关闭"入队但未注册"的窗口。
        """
        ack = threading.Event()
        self._store_queue.put(ack)
        if not ack.wait(timeout):
            logger.warning(
                "DiskBackend store_barrier timed out after %.0fs "
                "(store thread dead?)",
                timeout,
            )
            return False
        return True

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # 先排干 store 队列再发终止哨：滞后 store 的尾巴在引擎退出时仍
        # 在队列/线程中，直接杀 daemon 线程会截断 store 计数（体积账
        # 决策数与落盘数对不上）。屏障项排在既有批之后，被 set 时所有
        # 先入队批的 note_batch 已执行，计数闭合。超时兜底线程已死
        # （异常退出）的情况，此时尾巴本就保不住，只损失等待时间。
        self.store_barrier(timeout=120.0)
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
            if isinstance(item, threading.Event):
                # 队列屏障（store_barrier）：此刻先入队批次的 done 事件
                # 均已注册，set 放行等待中的 flush。
                item.set()
                continue
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            try:
                if wait_event is not None:
                    stream.wait_event(wait_event)
                    # 探针：主机侧等 compute_done 的耗时（依赖节拍税），
                    # 用于把 sync_wall 分解为 dep_wait + dma_wait。
                    if profiler.PROFILE:
                        _ts = profiler.now()
                    wait_event.synchronize()
                    if profiler.PROFILE:
                        profiler.note_dep_wait("store", profiler.now() - _ts)
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
        written = os.pwritev(
            self._fd, [self._store_slot_views[buf_slot]], file_offset
        )
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("store", written)

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        bytes_read = os.preadv(
            self._fd, [self._load_slot_views[buf_slot]], file_offset
        )
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("load", bytes_read)

    def _writev_range(self, buf_start: int, k: int, file_offset: int) -> None:
        """一次 pwritev 写出连续 k 个缓冲 slot（对应磁盘上的一段连续 run）。

        交错暂存缓冲下连续 k 个 slot 是单块连续内存 -> 1 条 iov。
        """
        ttb = self._total_block_bytes
        nbytes = k * ttb
        view = memoryview(
            self._store_np[buf_start * ttb : buf_start * ttb + nbytes]
        )
        written = os.pwritev(self._fd, [view], file_offset)
        if written < nbytes:
            raise OSError(
                f"Short write: expected {nbytes} bytes, wrote {written}"
            )
        if profiler.PROFILE:
            profiler.note_syscalls("store", written)

    def _readv_range(self, buf_start: int, k: int, file_offset: int) -> None:
        """一次 preadv 读入连续 k 个缓冲 slot（磁盘上一段连续 run）。"""
        ttb = self._total_block_bytes
        nbytes = k * ttb
        view = memoryview(
            self._load_np[buf_start * ttb : buf_start * ttb + nbytes]
        )
        bytes_read = os.preadv(self._fd, [view], file_offset)
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

        阶段二：提交时先按磁盘段重排 (gpu, slot) 对——多请求交错提交的
        slot 序列跨段跳动，但落盘互不依赖，段序重排让 run 合并看到
        段粒度的连续 slot（§3.7 的 74.6% 单块 run 主要来自交错而非空洞）。
        """
        assert self._store_params is not None
        half = self._coalesce_half
        # KVLog profiling 计时器（KVLOG_PROFILE=0 时为静态零开销）
        prof = profiler.PROFILE
        t0 = profiler.now() if prof else 0.0
        io_t = sync_t = dma_t = 0.0

        # 阶段二：段序重排（O(n log n)，纯 Python，~4 万块 < 10ms）
        gpu_blocks, disk_slots, _ = _reorder_by_segment(gpu_blocks, disk_slots)

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
                # 交错缓冲 stride != size，描述符合并恒不适用，逐块搬运
                coalesce=False,
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
                # 交错缓冲 stride != size，描述符合并恒不适用，逐块搬运
                coalesce=False,
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
