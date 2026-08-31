# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side handler for SimpleCPUOffloadConnector."""

from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.simple_kv_offload import profiler
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.disk_backend import DiskBackend
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class SimpleCPUOffloadWorker:
    """Worker-side handler for CPU offloading transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        kv_offload_backend: str = "cpu",
        disk_path: str | None = None,
        disk_capacity_bytes: int = 0,
        disk_buffer_slots: int = 2,
        use_page_cache: bool = False,
        disk_coalesce_io: bool = True,
        disk_segment_bytes: int = 16 * (1024**2),
        lag_store_steps: int = 0,
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        self.disk_path = disk_path
        self.disk_capacity_bytes = disk_capacity_bytes
        self.disk_buffer_slots = disk_buffer_slots
        self.use_page_cache = use_page_cache
        self.disk_coalesce_io = disk_coalesce_io
        self.disk_segment_bytes = disk_segment_bytes
        self.disk_mode = kv_offload_backend == "disk"
        # KVLog 第二步（滞后 store）：>0 时 store 批滞后 N 步提交（§4.6
        # 依赖节拍税的修复）。lag=1 为纯滞后；lag>=2 为跨步攒批（多步
        # 批合并为单批提交，同时摊薄交错碎片）。0 = 原生行为（A/B 基线）。
        self._lag_store_steps = max(0, int(lag_store_steps))
        # 持有中的滞后 store 批：(gpu_blocks, cpu_blocks, event_idx,
        # compute_done_event)。触发提交时只 flush 老化批并合并为单批，
        # 本步刚录事件的最新一批扣住不交（keep=1）。
        self._lagged_batches: list[
            tuple[list[int], list[int], int, torch.Event]
        ] = []

        self.gpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.cpu_kv_caches: dict[str, torch.Tensor] | None = None
        self.device: torch.device | None = None
        self.num_cpu_blocks: int = 0

        # CUDA streams for the async transfers
        self.load_stream: torch.cuda.Stream | None = None
        self.store_stream: torch.cuda.Stream | None = None

        self._backend: DmaCopyBackend | DiskBackend | None = None

        # Ordered (event_idx, Event). Events pre-allocated on main thread.
        self._load_events: list[tuple[int, torch.Event]] = []
        self._store_events: list[tuple[int, torch.Event]] = []
        # High-water marks: highest event_idx completed per stream.
        # When the event list is empty, the hwm covers all prior events.
        self._load_hwm: int = -1
        self._store_hwm: int = -1

        # Metadata for the current step
        self._connector_metadata: SimpleCPUOffloadMetadata | None = None

        # Compute-done event recorded before each store; reused across steps
        # (get_finished runs once per step, copy queue is FIFO).
        self._store_compute_done: torch.Event | None = None

        # Pending event index sets, populated in bind_connector_metadata
        self._pending_load_event_indices: set[int] = set()
        self._pending_store_event_indices: set[int] = set()
        # Completed store events to report via build_connector_worker_meta
        self._completed_store_events: dict[int, int] = {}

        # KVLog profiling（KVLOG_PROFILE=0 时不使用）：load 提交时间戳，
        # event_idx -> submit perf_counter
        self._load_submit_ts: dict[int, float] = {}

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ) -> None:
        """Register GPU KV caches and allocate pinned CPU tensors.
        The worker will infer the underlying raw storage from the kv_caches.

        Args:
            kv_caches: Per-layer GPU KV caches. Values are either a single
                tensor (attention layers) or a list of tensors (Mamba layers
                in hybrid models). All values are included for offloading
                by resolving to their underlying raw storage.
        """
        if not kv_caches:
            logger.warning("No KV caches to offload.")
            return

        self.device = next(iter(kv_caches.values())).device

        assert self.kv_cache_config is not None
        num_blocks = self.kv_cache_config.num_blocks

        # The DMA backend copies whole blocks as base + block_id * stride(0),
        # so view each unique allocation as [num_blocks, block_bytes].
        unique_gpu_caches: dict[str, torch.Tensor] = {}
        seen: set[tuple[torch.device, int]] = set()
        for name, tensor in kv_caches.items():
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if key in seen:
                continue
            seen.add(key)

            physical_per_block, remainder = divmod(tensor.shape[0], num_blocks)
            assert remainder == 0, (
                f"KV cache {name!r} has {tensor.shape[0]} physical blocks, which "
                f"is not divisible by {num_blocks} scheduler blocks"
            )
            block_bytes = tensor.stride(0) * tensor.element_size() * physical_per_block
            raw = torch.empty(0, dtype=torch.int8, device=tensor.device).set_(storage)
            regions = raw.view(-1, num_blocks, block_bytes)
            for idx, region in enumerate(regions):
                key_name = name if len(regions) == 1 else f"{name}.{idx}"
                unique_gpu_caches[key_name] = region

        # Compute per-tensor bytes_per_block. Tensors may have different
        # page_size_bytes (e.g., UniformTypeKVCacheSpecs with varying head_size).
        per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
        ]
        total_bytes_per_block = sum(per_tensor_bpb)

        self.num_cpu_blocks = max(1, self.cpu_capacity_bytes // total_bytes_per_block)

        # Use lowest priority so KV cache I/O yields to compute streams.
        low_pri, _ = torch.cuda.Stream.priority_range()
        self.load_stream = torch.cuda.Stream(priority=low_pri)
        self.store_stream = torch.cuda.Stream(priority=low_pri)

        self.gpu_kv_caches = unique_gpu_caches

        if self.disk_mode:
            self._init_disk_mode(unique_gpu_caches, total_bytes_per_block, self.device)
        else:
            self._init_cpu_mode(unique_gpu_caches, total_bytes_per_block, self.device)

    def _init_disk_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        num_disk_slots = max(1, self.disk_capacity_bytes // total_bytes_per_block)
        self.num_cpu_blocks = num_disk_slots

        logger.info(
            "SimpleCPUOffloadWorker [DISK]: %d tensors, %d disk slots (%.2f GB)",
            len(unique_gpu_caches),
            num_disk_slots,
            (num_disk_slots * total_bytes_per_block) / (1024**3),
        )

        assert self.disk_path is not None
        rank_path = f"{self.disk_path}.rank_{device.index or 0}"
        self._backend = DiskBackend()
        self._backend.init(
            unique_gpu_caches,
            device,
            self.load_stream,
            self.store_stream,
            rank_path,
            num_disk_slots,
            total_bytes_per_block,
            self.disk_buffer_slots,
            self.use_page_cache,
            coalesce_io=self.disk_coalesce_io,
            segment_bytes=self.disk_segment_bytes,
        )

    def _init_cpu_mode(
        self,
        unique_gpu_caches: dict[str, torch.Tensor],
        total_bytes_per_block: int,
        device: torch.device,
    ) -> None:
        logger.info(
            "SimpleCPUOffloadWorker [CPU]: %d tensors, %d CPU blocks (%.2f GB)",
            len(unique_gpu_caches),
            self.num_cpu_blocks,
            (self.num_cpu_blocks * total_bytes_per_block) / (1024**3),
        )

        pin_memory = PIN_MEMORY
        if not pin_memory:
            logger.warning(
                "Pinned memory not available. CPU offload performance may be degraded."
            )

        self.cpu_kv_caches = {}
        for name, gpu_tensor in unique_gpu_caches.items():
            cpu_shape = (self.num_cpu_blocks,) + gpu_tensor.shape[1:]
            # Allocate non-pinned first, then pin via cudaHostRegister to
            # bypass PyTorch's CUDACachingHostAllocator which rounds up to
            # the next power of 2 (e.g. 100 GB -> 128 GB).
            tensor = torch.zeros(cpu_shape, dtype=gpu_tensor.dtype, device="cpu")
            if pin_memory:
                pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor

        self._backend = DmaCopyBackend()
        self._backend.init(
            unique_gpu_caches,
            self.cpu_kv_caches,
            device,
            self.load_stream,
            self.store_stream,
        )

    def bind_connector_metadata(self, metadata: SimpleCPUOffloadMetadata) -> None:
        self._connector_metadata = metadata
        if metadata.load_event >= 0:
            self._pending_load_event_indices.add(metadata.load_event)
        if metadata.store_event >= 0:
            self._pending_store_event_indices.add(metadata.store_event)

    def clear_connector_metadata(self) -> None:
        self._connector_metadata = None

    def start_load_kv(self) -> None:
        # NOTE: we defer launching both load and store to get_finished(),
        # which runs after model execution. This hides the CPU-side
        # block copy op overhead (~5ms) behind GPU compute.
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Submit transfers and report completed events to the scheduler.

        Stores (GPU->CPU) read the live KV cache, which the compute stream may
        still be writing under v1 overlapped execution, so they are ordered
        after a compute-done event recorded on the current stream. Loads
        (CPU->GPU) read stable pinned host memory and launch immediately. See
        #45704 for the bug and #39306 for the srcAccessOrder rationale.

        Returns:
            tuple of (finished_sending, finished_recving).
            - finished_sending: always None (stores use worker metadata).
            - finished_recving: req_ids whose loads have completed.
        """
        # (1) Submit transfers
        metadata = self._connector_metadata
        if metadata is not None:
            backend = self._backend
            assert backend is not None
            if metadata.load_cpu_blocks:
                if profiler.PROFILE:
                    self._load_submit_ts[metadata.load_event] = profiler.now()
                backend.launch_copy(
                    metadata.load_cpu_blocks,
                    metadata.load_gpu_blocks,
                    is_store=False,
                    event_idx=metadata.load_event,
                    events_list=self._load_events,
                )
            if metadata.store_gpu_blocks:
                if self._lag_store_steps > 0:
                    # 滞后 store：每步新建 Event——持有跨步，不能复用
                    # self._store_compute_done（下一步 record 会重写同一
                    # Event，clobber 掉持有批的依赖）。
                    ev = torch.Event()
                    ev.record(torch.cuda.current_stream())
                    self._lagged_batches.append(
                        (
                            metadata.store_gpu_blocks,
                            metadata.store_cpu_blocks,
                            metadata.store_event,
                            ev,
                        )
                    )
                    if len(self._lagged_batches) > self._lag_store_steps:
                        # 只提交老化批：wait_event 必须至少滞后一步，否则
                        # store 线程仍等本步 forward，dep_wait 不会坍缩。
                        self._submit_lagged(backend, keep=1)
                else:
                    if self._store_compute_done is None:
                        self._store_compute_done = torch.Event()
                    self._store_compute_done.record(torch.cuda.current_stream())
                    backend.launch_copy(
                        metadata.store_gpu_blocks,
                        metadata.store_cpu_blocks,
                        is_store=True,
                        event_idx=metadata.store_event,
                        events_list=self._store_events,
                        wait_event=self._store_compute_done,
                    )
            elif self._lagged_batches:
                # 本步无新 store：立即排空持有批（步间隙/尾部排空，
                # 避免最后一批滞留到引擎关闭）
                self._submit_lagged(backend)

        # (2) Track completed transfer events
        finished_recving: set[str] = set()

        if profiler.PROFILE:
            profiler.note_pending(len(self._pending_load_event_indices))

        if self._pending_load_event_indices:
            load_wm = self._poll_stream_events(is_store=False)
            for j in [j for j in self._pending_load_event_indices if j <= load_wm]:
                self._pending_load_event_indices.discard(j)
                req_ids = (
                    metadata.load_event_to_reqs.get(j) if metadata is not None else None
                )
                if req_ids:
                    finished_recving.update(req_ids)

        if self._pending_store_event_indices:
            store_wm = self._poll_stream_events(is_store=True)
            for j in [j for j in self._pending_store_event_indices if j <= store_wm]:
                self._pending_store_event_indices.discard(j)
                self._completed_store_events[j] = 1

        return None, finished_recving or None

    def build_connector_worker_meta(self) -> SimpleCPUOffloadWorkerMetadata | None:
        """Return completed store events since the last call."""
        if not self._completed_store_events:
            return None
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events=self._completed_store_events,
        )
        self._completed_store_events = {}
        return meta

    def _submit_lagged(
        self, backend: DmaCopyBackend | DiskBackend, keep: int = 0
    ) -> None:
        """提交持有中的滞后 store 批（老化批合并为单批提交）。

        keep>0 时扣住最新 keep 批不提交：其 compute_done 是本步刚录的
        事件，若随批交出，store 线程仍要等本步 forward 跑完——与原生
        路径同样的依赖节拍税。只提交至少滞后一步的老化批，提交时刻
        事件通常已完成，dep_wait 才可能坍缩（RQ3b 锚点）。

        合并批的 wait_event 取被提交批中最后一个：compute 流按序执行，
        后录事件蕴含此前全部依赖，一次等待覆盖所有被提交批。event_idx
        取被提交批的最大值：store 完成回报按水位（hwm）推进，完成时
        所有 <= 水位的 pending 索引一并标记完成，调度侧块释放语义不变。

        正确性：滞后只推迟搬运时机，不改数据所有权——块释放本就门在
        store 完成回执上，持有期间不会被复用；代价是换出窗口内 KV
        驻留 GPU 多 lag_store_steps 步（内存压力场景需评估，RQ4 消融）。
        """
        if keep >= len(self._lagged_batches):
            return
        flush = self._lagged_batches[: len(self._lagged_batches) - keep]
        self._lagged_batches = self._lagged_batches[
            len(self._lagged_batches) - keep :
        ]
        if len(flush) == 1:
            gpu_blocks, cpu_blocks, event_idx, ev = flush[0]
        else:
            gpu_blocks = [b for bs, _, _, _ in flush for b in bs]
            cpu_blocks = [s for _, ss, _, _ in flush for s in ss]
            event_idx = flush[-1][2]
            ev = flush[-1][3]
        backend.launch_copy(
            gpu_blocks,
            cpu_blocks,
            is_store=True,
            event_idx=event_idx,
            events_list=self._store_events,
            wait_event=ev,
        )

    def handle_preemptions(
        self, kv_connector_metadata: SimpleCPUOffloadMetadata
    ) -> None:
        """Sync all in-flight transfers before preempted blocks are reused."""
        if not kv_connector_metadata.need_flush:
            return
        self._flush_and_sync_all()

    def _flush_and_sync_all(self) -> None:
        """Synchronize all in-flight transfer events."""
        # 滞后 store：先把持有批提交出去再同步——抢占后块可能被复用，
        # 持有批的 DMA 必须在 flush 返回前完成，否则读到复用后的数据。
        if self._lagged_batches and self._backend is not None:
            self._submit_lagged(self._backend)
            # launch_copy 只入队，done 事件要等 store 线程处理到该队列项
            # 之后才注册；不先过队列屏障，下面的事件同步覆盖不到持有批。
            self._backend.store_barrier()
        # KVLog profiling：抢占 flush 是 engine 的真实阻塞点
        t0 = profiler.now() if profiler.PROFILE else 0.0
        for event_idx, event in self._load_events:
            event.synchronize()
            self._load_hwm = event_idx
        self._load_events.clear()

        for event_idx, event in self._store_events:
            event.synchronize()
            self._store_hwm = event_idx
        self._store_events.clear()
        if profiler.PROFILE:
            profiler.note_flush(profiler.now() - t0)

    def _poll_stream_events(self, is_store: bool) -> int:
        """Non-blocking poll for completed events and return the high-water mark."""
        events = self._store_events if is_store else self._load_events
        hwm = self._store_hwm if is_store else self._load_hwm
        while events:
            event_idx, event = events[0]
            if not event.query():
                break
            hwm = event_idx
            events.pop(0)
            # KVLog profiling：load 完成延迟（submit -> done）
            if profiler.PROFILE and not is_store:
                submit_ts = self._load_submit_ts.pop(event_idx, None)
                if submit_ts is not None:
                    profiler.note_load_event(submit_ts, profiler.now())
        if is_store:
            self._store_hwm = hwm
        else:
            self._load_hwm = hwm
        return hwm
