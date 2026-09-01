# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side manager for SimpleCPUOffloadConnector."""

import contextlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload import profiler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class TransferMeta:
    gpu_block_ids: list[int]
    cpu_block_ids: list[int]


@dataclass
class LoadRequestState:
    request: "Request"
    transfer_meta: TransferMeta
    load_event: int | None = None
    finished: bool = False


# NOTE: This per-request state is only used in eager mode.
@dataclass
class StoreRequestState:
    request: "Request"
    # Accumulated block IDs from scheduler_output via yield_req_data.
    block_ids: tuple[list[int], ...]
    # Per-group cursors tracking how many blocks have been stored/skipped.
    num_stored_blocks: list[int]
    store_events: set[int] = field(default_factory=set)
    # WriteGate v2：被暂缓的块 (group_idx, gpu_block_id, position) 重试队列。
    # 游标照常推进（语义=已决策过）；每步扫描前先重试 pending，
    # ref_cnt 被兄弟请求顶上去后翻盘写盘。请求被抢占时块 id 失效，
    # 队列随 block_ids 一起清空。position 是该块在本请求 block_ids[g] 中的
    # 下标，供 v3 的 ledger 判据回溯祖先链。
    gate_pending: list[tuple[int, int, int]] = field(default_factory=list)
    # WriteGate v3 ledger：前缀家族复用前沿（见 _gate_family_hit）。
    # ledger_front[g] = 已确认"自身与祖先都未被读回过"的前缀长度；
    # ledger_hit_at[g] = 第一个"曾被读回"的块下标（-1 = 未发现）。
    # 命中是单调事实，故前沿只前进不后退，判据摊还 O(1)。
    ledger_front: list[int] = field(default_factory=list)
    ledger_hit_at: list[int] = field(default_factory=list)
    finished: bool = False


class _DiskSegmentAllocator:
    """段式磁盘 slot 分配器（KVLog M3 阶段一）。

    - 固定 ``segment_size``（默认 32）slot 段；64 KiB block 下即 2 MiB；
    - 段内 bump 顺序分配：同一请求先后落盘的块天然聚成物理连续 run，
      是 disk_backend run 合并 I/O（单次大 pwritev/preadv）的前提；
    - 整段空闲时归还段空闲列表（FIFO 轮转，最大化旧缓存数据存活时间）；
      阶段一简单版不做段内整理，部分空闲的段不参与再分配；
    - 以 BlockPool 的 ``ref_cnt`` 为唯一事实源：每次分发前逐 slot 校验，
      被 load pin 或其他路径占用的 slot 自动跳过，任何分配路径都不会
      造成双重分配。
    """

    def __init__(
        self,
        pool: BlockPool,
        num_slots: int,
        segment_size: int = 32,
    ) -> None:
        self._pool = pool
        self._blocks = pool.blocks
        self._num_slots = num_slots
        self._seg_size = segment_size
        self._num_segs = cdiv(num_slots, segment_size)
        self._free_segs: deque[int] = deque(range(self._num_segs))
        self._seg_free: list[bool] = [True] * self._num_segs
        self._active: int = -1
        self._off: int = 0
        # 阶段二段亲和：req_key -> (最近分配的段, 段内下一偏移)
        self._affinity: dict[str, tuple[int, int]] = {}
        # 观测计数（段利用率计量用）
        self.num_taken: int = 0
        self.num_segments_recycled: int = 0

    def take_block(self) -> "KVCacheBlock | None":
        """段内 bump 取下一个空闲 slot，并完成池记账（出队/逐 hash/引用计数）。

        容量耗尽（无整段空闲可激活）时返回 None，调用方按 out_of_space 处理。
        """
        pool = self._pool
        while True:
            if self._active < 0:
                if not self._free_segs:
                    return None
                self._active = self._free_segs.popleft()
                self._seg_free[self._active] = False
                self._off = 0
            base = self._active * self._seg_size
            while self._off < self._seg_size:
                bid = base + self._off
                self._off += 1
                if bid >= self._num_slots:
                    break
                blk = self._blocks[bid]
                if blk.is_null or blk.ref_cnt != 0:
                    # 被 load pin 或其他分配路径占用，跳过保正确性
                    continue
                pool.free_block_queue.remove(blk)
                if pool.enable_caching:
                    pool._maybe_evict_cached_block(blk)
                blk.ref_cnt += 1
                if pool.metrics_collector is not None:
                    pool.metrics_collector.on_block_allocated(blk)
                self.num_taken += 1
                return blk
            # 当前段已扫完，换下一段；若整段已空闲（如块在 active 期间
            # 被释放且之后再无该段的释放事件），立即归还避免槽位滞留
            seg = self._active
            self._active = -1
            self._try_recycle(seg)

    def take_block_affinity(self, req_key: str) -> "KVCacheBlock | None":
        """阶段二段亲和分配：同 req_key 的块优先聚到其上次分配的段。

        lazy 模式下多请求交错调用 take_block 会把段内连续性切碎
        （§3.7：74.6% 单块 run）。此变体为每个 req_key 记住最近分配的
        （段, 段内偏移），后续同 key 的块优先回到该段继续 bump——
        段内空洞（被其他请求占用/pin 的 slot）自然跳过，块间连续性
        尽力保持。耗尽或段被回收时回退到 take_block 的全局 bump。
        """
        blk = self._take_from_affinity(req_key)
        if blk is not None:
            self.num_taken += 1
            return blk
        return self.take_block()

    def _take_from_affinity(self, req_key: str) -> "KVCacheBlock | None":
        """尝试从 req_key 的亲和段内取块；段满/无效时清除亲和返回 None。"""
        state = self._affinity.get(req_key)
        if state is None:
            return None
        seg, off = state
        base = seg * self._seg_size
        if seg >= self._num_segs or self._seg_free[seg]:
            # 段已被整体回收（或越界），亲和失效
            self._affinity.pop(req_key, None)
            return None
        pool = self._pool
        while off < self._seg_size:
            bid = base + off
            off += 1
            if bid >= self._num_slots:
                break
            blk = self._blocks[bid]
            if blk.is_null or blk.ref_cnt != 0:
                continue
            pool.free_block_queue.remove(blk)
            if pool.enable_caching:
                pool._maybe_evict_cached_block(blk)
            blk.ref_cnt += 1
            if pool.metrics_collector is not None:
                pool.metrics_collector.on_block_allocated(blk)
            self._affinity[req_key] = (seg, off)
            return blk
        # 亲和段已满：清除亲和，回退全局 bump（会开新段并重建亲和）
        self._affinity.pop(req_key, None)
        return None

    def _try_recycle(self, seg: int) -> None:
        """段内全部 slot 空闲（ref_cnt==0 或 null）时归还段空闲列表尾部。"""
        if self._seg_free[seg]:
            return
        base = seg * self._seg_size
        end = min(base + self._seg_size, self._num_slots)
        if all(
            self._blocks[b].is_null or self._blocks[b].ref_cnt == 0
            for b in range(base, end)
        ):
            self._seg_free[seg] = True
            self._free_segs.append(seg)
            self.num_segments_recycled += 1

    def note_freed(self, block_ids: Iterable[int]) -> None:
        """块归还后检查所属段是否整段空闲，是则归还段空闲列表尾部。"""
        for seg in {bid // self._seg_size for bid in block_ids}:
            if seg == self._active:
                continue
            self._try_recycle(seg)


# --- WriteGate v3：闭环反馈回路（在线死写率 -> 准入激进度）----------------

GATE_RELAX = "relax"
GATE_MID = "mid"
GATE_STRICT = "strict"


class WriteGateController:
    """按在线死写率调节写准入激进度的反馈控制器（提案 §5.3 待接项）。

    信号源与论文体积账**同一套计数**（profiler 的写侧账本 + 读回命中表）：
    写出去的块在一个成熟窗（horizon，按后续写盘块数计——约 1/3 池周转的
    量级）内始终没被读回过，即结算为一次死写。控制器在结算结果的滑动窗上
    算死写率，带迟滞地在三档间迁移，一次只走一档：

    - ``relax``   ：全写（等价 gate 关闭）。冷启动档位——没有浪费证据之前
                    不丢任何一次写，避免误杀首屏复用。
    - ``mid``     ：share ∨ ledger。当下被多请求共享，或该前缀家族历史上
                    被读回过，才写。
    - ``strict``  ：ledger ∨ lifecycle。share 在冷回放下与未来复用负相关
                    （v1 实测 100% 死写），证据不足时只信历史复用记录。

    迟滞带：升档 0.50 / 0.90，降档 0.45 / 0.75，且需攒够 ``eval_every``
    个成熟样本才允许改档，避免阈值附近抖动让写量来回摆。lifecycle 保底
    （被抢占请求的块无条件写）不经过控制器，任何档位都保留。

    观测窗冻结保护：拒写太彻底就没有新块可结算、也就没有新证据。若连续
    ``freeze_steps`` 步在拒写但成熟数为 0，则主动降一档重新探索，保证回路
    不会单向棘死在 strict。只放开 **strict -> mid** 这一条冻结通道：mid 冻
    结意味着"连当下共享都没有"，再退到 relax 等于无条件全写，只会把刚省掉
    的死写重新堆回来；回到 relax 必须由证据（低死写率）驱动。
    """

    # 升/降档阈值（中间是迟滞带）
    UP_TO_MID = 0.50
    UP_TO_STRICT = 0.90
    DOWN_TO_MID = 0.75
    DOWN_TO_RELAX = 0.45

    def __init__(
        self,
        horizon_blocks: int = 4096,
        window_blocks: int = 2048,
        eval_every_blocks: int = 2048,
        freeze_steps: int = 200,
    ) -> None:
        self.horizon = horizon_blocks
        self.eval_every = eval_every_blocks
        self.freeze_steps = freeze_steps
        # 待成熟队列：(提交序号, 块哈希)
        self._ring: deque[tuple[int, bytes]] = deque()
        # 成熟结算结果：1 = 死写（越过 horizon 仍无命中），0 = 活
        self._verdicts: deque[int] = deque(maxlen=window_blocks)
        self._seq = 0
        self._due = 0
        self._frozen = 0
        self.tier = GATE_RELAX
        self.dead_rate = 0.0
        self.matured_total = 0
        self.dead_total = 0
        self.transitions = 0
        self.frozen_downgrades = 0
        self._seen_drops = 0
        self._tiers_seen: set[str] = {GATE_RELAX}
        self._trajectory: list[dict] = []
        self._max_trajectory = 64

    # --- 回路入口：每次写准入扫描后调用一次 ---------------------------------

    def note_step(self, stored_hashes: list[bytes], n_dropped: int) -> None:
        """喂入本步写盘块与本步拒写块数，推进成熟窗并（按节流）调档。"""
        for h in stored_hashes:
            self._ring.append((self._seq, h))
            self._seq += 1
        self._seen_drops += n_dropped

        cutoff = self._seq - self.horizon
        matured = 0
        while self._ring and self._ring[0][0] < cutoff:
            _, h = self._ring.popleft()
            dead = 0 if profiler.was_ever_hit(h) else 1
            self._verdicts.append(dead)
            self.matured_total += 1
            self.dead_total += dead
            self._due += 1
            matured += 1

        if matured or self._seen_drops == 0 or self.tier == GATE_RELAX:
            self._frozen = 0
        else:
            self._frozen += 1

        if self._due >= self.eval_every:
            self._due = 0
            self._rebalance()
        elif self._frozen >= self.freeze_steps:
            self._frozen = 0
            self._downgrade_frozen()

    # --- 档位迁移 -----------------------------------------------------------

    def _rebalance(self) -> None:
        if len(self._verdicts) < self.eval_every:
            return  # 样本不足，保持当前档
        rate = sum(self._verdicts) / len(self._verdicts)
        self.dead_rate = rate
        if self.tier == GATE_RELAX:
            if rate >= self.UP_TO_MID:
                self._move(GATE_MID, rate, "dead_rate")
        elif self.tier == GATE_MID:
            if rate >= self.UP_TO_STRICT:
                self._move(GATE_STRICT, rate, "dead_rate")
            elif rate <= self.DOWN_TO_RELAX:
                self._move(GATE_RELAX, rate, "dead_rate")
        elif rate <= self.DOWN_TO_MID:
            self._move(GATE_MID, rate, "dead_rate")

    def _downgrade_frozen(self) -> None:
        """成熟窗冻结（持续拒写、无新证据）时主动降一档探索。

        仅 strict -> mid：mid 已含 share 信号，再冻结说明当下无任何共享可
        依，退到 relax 只是无条件全写，会把省掉的死写重新堆回来。
        """
        if self.tier == GATE_STRICT:
            self.frozen_downgrades += 1
            self._move(GATE_MID, self.dead_rate, "window_frozen")

    def _move(self, tier: str, rate: float, reason: str) -> None:
        prev = self.tier
        self.tier = tier
        self.transitions += 1
        self._tiers_seen.add(tier)
        if len(self._trajectory) < self._max_trajectory:
            self._trajectory.append(
                {
                    "at_stored": self._seq,
                    "dead_rate": round(rate, 4),
                    "tier": tier,
                    "reason": reason,
                }
            )
        logger.info(
            "WriteGate auto: %s -> %s (在线死写率 %.3f, 成熟窗 %d 块, "
            "已写 %d 块, %s)",
            prev,
            tier,
            rate,
            len(self._verdicts),
            self._seq,
            reason,
        )

    # --- 可观测性 -----------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "tier": self.tier,
            "tiers_seen": sorted(self._tiers_seen),
            "transitions": self.transitions,
            "frozen_downgrades": self.frozen_downgrades,
            "dead_rate_online": round(self.dead_rate, 4),
            "window_size": len(self._verdicts),
            "matured_blocks": self.matured_total,
            "dead_blocks_matured": self.dead_total,
            "stored_seen": self._seq,
            "dropped_seen": self._seen_drops,
            "horizon_blocks": self.horizon,
            "eval_every_blocks": self.eval_every,
            "trajectory": list(self._trajectory),
        }


class SimpleCPUOffloadScheduler:
    """Scheduler-side manager for CPU offloading."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: "KVCacheConfig | None",
        cpu_capacity_bytes: int,
        scheduler_block_size: int,
        hash_block_size: int,
        lazy_offload: bool = False,
        disk_capacity_bytes: int = 0,
        write_gate_signals: str = "",
    ):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # When disk mode is active, the offload pool size is disk-based.
        offload_capacity = (
            disk_capacity_bytes if disk_capacity_bytes > 0 else cpu_capacity_bytes
        )
        self.enable_kv_cache_events = (
            vllm_config.kv_events_config is not None
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.cp_world_size = dcp_world_size
        self.block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        assert self.block_size % self.hash_block_size == 0
        # Derive a CPU KVCacheConfig from the GPU config and build a coordinator
        assert kv_cache_config is not None
        self.cpu_kv_cache_config = self._derive_cpu_config(
            kv_cache_config, offload_capacity
        )
        self.num_cpu_blocks = self.cpu_kv_cache_config.num_blocks
        # Find the full attention kv group for prefix cache matching.
        self.fa_gidx = -1
        for g_idx, g in enumerate(self.cpu_kv_cache_config.kv_cache_groups):
            if isinstance(g.kv_cache_spec, FullAttentionSpec):
                self.fa_gidx = g_idx
                break
        assert 0 <= self.fa_gidx < len(self.cpu_kv_cache_config.kv_cache_groups)
        # FA group's own block_size; divides scheduler_block_size (the LCM)
        # but is NOT assumed to equal it.
        self.fa_block_size: int = (
            self.cpu_kv_cache_config.kv_cache_groups[
                self.fa_gidx
            ].kv_cache_spec.block_size
            * self.cp_world_size
        )
        assert self.block_size % self.fa_block_size == 0

        logger.info(
            "SimpleCPUOffloadScheduler: Allocating %d offload blocks "
            "(%.2f GB, mode=%s, backend=%s)",
            self.num_cpu_blocks,
            offload_capacity / (1024**3),
            "lazy" if lazy_offload else "eager",
            "disk" if disk_capacity_bytes > 0 else "cpu",
        )

        # WriteGate v1：写准入信号集（空 = 关闭，原生全写基线）。
        # share 走 BlockPool.ref_cnt（请求分配/前缀命中各 +1，radix 注册
        # 不持引用），故 ref_cnt>1 = 前缀主干被多请求共享，ref_cnt==1
        # = 单请求私有尾缀。lifecycle = 被抢占请求无条件写（恢复省重算）。
        self._write_gate_signals: set[str] = {
            s.strip() for s in write_gate_signals.split(",") if s.strip()
        }
        # WriteGate v2：首丢 hash 集合（体积账去重键；hash 内容寻址稳定）
        self._gate_dropped_hashes: set[bytes] = set()
        # WriteGate v3：ledger（历史复用账本）与 auto（闭环控制器）都以
        # profiler 的读回命中表为唯一数据源。profiler 未开时该表恒空，
        # "无数据"会被误判成"无命中"而把写全部拒掉，故此处显式降级并告警。
        self._gate_ledger = bool(self._write_gate_signals & {"ledger", "auto"})
        self._gate_ctrl: WriteGateController | None = None
        if self._gate_ledger and not profiler.PROFILE:
            logger.warning(
                "WriteGate signals %s 依赖 kvlog_profile 在线账本，但 profiler "
                "未激活 -> 丢弃 ledger/auto 信号，退化为 %s",
                sorted(self._write_gate_signals),
                sorted(self._write_gate_signals - {"ledger", "auto"}) or "关闭",
            )
            self._write_gate_signals -= {"ledger", "auto"}
            self._gate_ledger = False
        if "auto" in self._write_gate_signals:
            self._gate_ctrl = WriteGateController()
        if self._write_gate_signals:
            logger.info(
                "SimpleCPUOffloadScheduler: WriteGate signals=%s "
                "(eager store path only)%s",
                sorted(self._write_gate_signals),
                (
                    ""
                    if self._gate_ctrl is None
                    else (
                        " auto[closed-loop] start=%s horizon=%d eval_every=%d "
                        "up=%.2f/%.2f down=%.2f/%.2f"
                        % (
                            self._gate_ctrl.tier,
                            self._gate_ctrl.horizon,
                            self._gate_ctrl.eval_every,
                            self._gate_ctrl.UP_TO_MID,
                            self._gate_ctrl.UP_TO_STRICT,
                            self._gate_ctrl.DOWN_TO_RELAX,
                            self._gate_ctrl.DOWN_TO_MID,
                        )
                    )
                ),
            )
            if lazy_offload:
                logger.warning(
                    "WriteGate signals are only applied on the eager store "
                    "path; lazy mode keeps native all-write behavior."
                )

        spec_config = vllm_config.speculative_config
        use_eagle = spec_config is not None and spec_config.use_eagle()
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
            kv_cache_config=self.cpu_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=True,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=1,
            scheduler_block_size=self.block_size,
            hash_block_size=self.hash_block_size,
        )
        self.cpu_block_pool: BlockPool = self.cpu_coordinator.block_pool
        # GPU block pool reference - bound after scheduler builds kv_cache_manager
        self._gpu_block_pool: BlockPool | None = None

        # KVLog M3 阶段一：磁盘模式下用段式分配器保证 disk slot 物理连续
        # （run 合并 I/O 的前提）。CPU 模式保持原生 free-queue 分配。
        self._disk_seg_alloc: _DiskSegmentAllocator | None = None
        if disk_capacity_bytes > 0:
            self._disk_seg_alloc = _DiskSegmentAllocator(
                self.cpu_block_pool, self.num_cpu_blocks, segment_size=32
            )
            logger.info(
                "SimpleCPUOffloadScheduler: disk segment allocator on "
                "(%d slots, seg=32)",
                self.num_cpu_blocks,
            )

        # Load metadata
        self._reqs_to_load: dict[str, LoadRequestState] = {}
        # Inverse map: load_event_idx -> req_ids. Keyed by load_event_idx because
        # the worker reports completions by event index, not request id.
        self._load_event_to_reqs: dict[int, list[str]] = {}

        # Pending (cpu_hit_blocks, hit_length) tuples from find_longest_cache_hit,
        # kept pinned via touch() while awaiting update_state_after_alloc().
        self._pending_cpu_hits: dict[
            str, tuple[tuple[list[KVCacheBlock], ...], int]
        ] = {}

        # Store metadata
        self._lazy_mode = lazy_offload
        # Lazy mode: use a cursor to track the last scanned block in the GPU free queue.
        self._cursor: KVCacheBlock | None = None
        if self._lazy_mode:
            self._target_free = self._estimate_lazy_target_blocks(
                kv_cache_config,
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.cp_world_size,
            )
        else:
            self._target_free = 0
        self._store_event_to_blocks: dict[int, TransferMeta] = {}
        self._abandoned_store_event_to_blocks: dict[int, TransferMeta] = {}
        # Eager mode only
        self._reqs_to_store: dict[str, StoreRequestState] = {}
        self._store_event_to_reqs: dict[int, list[str]] = {}
        self._in_flight_store_gpu_blocks: set[int] = set()
        self._abandoned_reqs_to_load: dict[str, LoadRequestState] = {}

        # Event counters
        self._load_event_counter: int = 0
        self._store_event_counter: int = 0

        # For TP/PP: track partial store completions across steps.
        # Events must be reported by all world_size workers before considered complete.
        self._expected_worker_count = vllm_config.parallel_config.world_size
        self._store_event_pending_counts: dict[int, int] = {}

    @staticmethod
    def _derive_cpu_config(
        gpu_config: "KVCacheConfig", cpu_capacity_bytes: int
    ) -> "KVCacheConfig":
        """Derive a CPU KVCacheConfig from the GPU config.
        Same kv_cache_groups, num_blocks scaled by CPU/GPU memory ratio."""
        # Import here to avoid potential circular imports
        from vllm.v1.kv_cache_interface import KVCacheTensor

        assert len(gpu_config.kv_cache_tensors) > 0

        # Every KVCacheTensor describes placement within the same backing allocation,
        # so its size is the total GPU KV cache size.
        gpu_total_bytes = gpu_config.kv_cache_tensors[0].size
        num_gpu_blocks = gpu_config.num_blocks
        num_cpu_blocks = max(1, num_gpu_blocks * cpu_capacity_bytes // gpu_total_bytes)
        # Create CPU kv_cache_tensors mirroring GPU by scaling size proportionally.
        cpu_tensors = [
            KVCacheTensor(
                size=t.size // num_gpu_blocks * num_cpu_blocks,
                layers=list(t.layers),
                layer_stride=t.layer_stride,
                block_stride=t.block_stride,
                offset=t.offset,
            )
            for t in gpu_config.kv_cache_tensors
        ]

        return replace(
            gpu_config,
            num_blocks=num_cpu_blocks,
            kv_cache_tensors=cpu_tensors,
        )

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        cp_world_size: int = 1,
    ) -> int:
        """GPU blocks to keep available (free/offloaded) per step in lazy mode."""
        WATERMARK_RATIO = 1.0  # Reserve larger space to avoid running out of GPU blocks
        target = 0
        for g in kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            block_size = spec.block_size * cp_world_size
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return int(target * (1 + WATERMARK_RATIO))

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        """Bind GPU block pool so that we can touch blocks during stores.
        Called by Scheduler after kv_cache_manager is ready."""
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Return (num_new_tokens, is_async) from consecutive CPU cache hits."""

        # Pins found CPU blocks so they survive LRU eviction until
        # update_state_after_alloc() consumes them. Any pin from an earlier
        # call on the same request (e.g. retry after a failed allocate_slots)
        # is dropped first.
        if stale := self._pending_cpu_hits.pop(request.request_id, None):
            self._free_pending_cpu_hit(stale)

        num_skipped_hashes = num_computed_tokens // self.hash_block_size
        remaining_hashes = request.block_hashes[num_skipped_hashes:]

        if not remaining_hashes:
            return 0, False
        # Must recompute at least the last token, matching the logic in
        # kv_cache_manager.get_computed_blocks().
        max_hit_len = request.num_tokens - 1 - num_computed_tokens
        if max_hit_len <= 0:
            return 0, False
        cpu_hit_blocks, hit_length, _ = self.cpu_coordinator.find_longest_cache_hit(
            remaining_hashes, max_hit_len
        )

        if hit_length > 0:
            pin_blocks = [
                blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
            ]
            self.cpu_block_pool.touch(pin_blocks)
            self._pending_cpu_hits[request.request_id] = (
                cpu_hit_blocks,
                hit_length,
            )
            return hit_length, True
        return 0, False

    # TODO(yifan): this API now only matches the suffix part of the prefix cache. A more
    # general API should scan blocks in both GPU and CPU block pool in a single pass.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        block_ids_by_group = blocks.get_block_ids()
        num_groups = len(block_ids_by_group)

        # Store tracking (eager mode only). Register the request;
        # block IDs are accumulated from scheduler_output in
        # _prepare_eager_store_specs via yield_req_data.
        if not self._lazy_mode and req_id not in self._reqs_to_store:
            self._reqs_to_store[req_id] = StoreRequestState(
                request=request,
                block_ids=tuple([] for _ in range(num_groups)),
                num_stored_blocks=[0] * num_groups,
                ledger_front=[0] * num_groups,
                ledger_hit_at=[-1] * num_groups,
            )

        # Pop the CPU hit cached by get_num_new_matched_tokens(). The
        # found blocks were pinned there to survive LRU eviction in the window
        # between get_num_new_matched_tokens() and this matching call.
        pending = self._pending_cpu_hits.pop(req_id, None)

        if num_external_tokens == 0:
            if pending is not None:
                logger.warning(
                    "SimpleCPUOffloadScheduler: update_state_after_alloc "
                    "called for req_id=%s with no external tokens but "
                    "get_num_new_matched_tokens() unexpectedly recorded "
                    "a pending CPU hit; releasing the stale pin.",
                    req_id,
                )
                self._free_pending_cpu_hit(pending)
            return

        if pending is None:
            logger.warning(
                "SimpleCPUOffloadScheduler: update_state_after_alloc called "
                "for req_id=%s with num_external_tokens=%d but no pending "
                "CPU hit from get_num_new_matched_tokens(); skipping load.",
                req_id,
                num_external_tokens,
            )
            return

        cpu_hit_blocks_full, _ = pending

        # ``num_external_tokens`` is LCM-aligned (checked per-group below),
        # so this counts whole scheduler-aligned chunks of incoming tokens.
        num_blocks_to_load = num_external_tokens // self.block_size
        assert num_blocks_to_load > 0
        num_cached_fa_blocks = sum(
            blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx]
        )
        num_computed_tokens = num_cached_fa_blocks * self.fa_block_size

        # Build transfer pairs across all groups.
        total_computed_tokens = num_computed_tokens + num_external_tokens
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups

        # The scheduler may have accepted fewer blocks than
        # get_num_new_matched_tokens() reported.
        # (e.g. due to token budget in test_partial_gpu_prefix_plus_cpu_load).
        # Take only the leading N blocks per group matching num_external_tokens;
        # the rest will be released along with the temp pin below.
        cpu_hit_blocks: list[list[KVCacheBlock]] = []
        for g in range(num_groups):
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            assert num_external_tokens % g_block_size == 0, (
                f"num_external_tokens={num_external_tokens} not aligned to "
                f"group {g} block_size={g_block_size}"
            )
            n_take_g = num_external_tokens // g_block_size
            cpu_hit_blocks.append(cpu_hit_blocks_full[g][:n_take_g])

        gpu_block_ids: list[int] = []
        cpu_block_ids: list[int] = []
        cpu_blocks_to_touch: list[KVCacheBlock] = []
        load_hit_hashes: list[bytes] = []

        for g in range(num_groups):
            cpu_blocks_g = cpu_hit_blocks[g]
            n_ext_g = len(cpu_blocks_g)
            if n_ext_g == 0:
                continue

            # Number of blocks in the computed range for this group.
            g_block_size = (
                kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
            )
            n_computed_g = cdiv(total_computed_tokens, g_block_size)

            # Back-trace: ext blocks sit at the tail of the computed range.
            gpu_ext_start = n_computed_g - n_ext_g
            group_gpu_ids = block_ids_by_group[g]

            for i, cpu_blk in enumerate(cpu_blocks_g):
                # Skip null blocks (e.g. sliding window or mamba padding).
                if cpu_blk.is_null:
                    continue
                gpu_block_ids.append(group_gpu_ids[gpu_ext_start + i])
                cpu_block_ids.append(cpu_blk.block_id)
                cpu_blocks_to_touch.append(cpu_blk)
                if profiler.PROFILE and cpu_blk.block_hash is not None:
                    load_hit_hashes.append(cpu_blk.block_hash)

        if profiler.PROFILE and load_hit_hashes:
            # 体积账：读回侧每块命中次数 +1，与写侧对账死写。
            profiler.note_block_loads(load_hit_hashes)

        # Touch CPU blocks to prevent eviction during async load.
        self.cpu_block_pool.touch(cpu_blocks_to_touch)
        # Release the temporary pin held since get_num_new_matched_tokens().
        self._free_pending_cpu_hit(pending)

        # Touch GPU blocks to prevent freeing during async load
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
        )

        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request, transfer_meta=TransferMeta(gpu_block_ids, cpu_block_ids)
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        if store_gpu:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            if store_req_ids:  # For eager mode only, track req->blocks mapping
                self._store_event_to_reqs[store_event] = store_req_ids
                for req_id in store_req_ids:
                    store_state = self._reqs_to_store.get(req_id)
                    if store_state is not None:
                        store_state.store_events.add(store_event)

        # --- Loads ---
        load_event = -1
        load_gpu: list[int] = []
        load_cpu: list[int] = []
        load_req_ids: list[str] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            load_req_ids.append(req_id)
        if load_req_ids:
            load_event = self._load_event_counter
            self._load_event_counter += 1
            for req_id in load_req_ids:
                self._reqs_to_load[req_id].load_event = load_event
            self._load_event_to_reqs[load_event] = load_req_ids

        result = SimpleCPUOffloadMetadata(
            load_event=load_event,
            load_gpu_blocks=load_gpu,
            load_cpu_blocks=load_cpu,
            load_event_to_reqs={
                event_idx: list(req_ids)
                for event_idx, req_ids in self._load_event_to_reqs.items()
            },
            store_event=store_event,
            store_gpu_blocks=store_gpu,
            store_cpu_blocks=store_cpu,
            need_flush=bool(scheduler_output.preempted_req_ids),
        )
        return result

    def _gate_family_hit(self, state: StoreRequestState, g: int, pos: int) -> bool:
        """ledger 判据：该块自身或其**祖先**是否曾被 offload 读回过。

        直觉：一个前缀家族历史上被复用，说明这类 prompt 是"活的"，其后缀
        写盘大概率不会白写（多轮对话、共享 system prompt 的负载结构）。
        与 v1 的 share 区别在于这是**历史证据**而非决策时点的瞬时观测。

        命中是单调事实（profiler 只 +1 不清零），"某个祖先被命中"对 pos
        也是单调的——一旦前沿右侧出现命中，其右侧所有块都判 True。因此用
        per-(req, group) 的前沿游标只做单向扫描，摊还 O(1)。
        已知局限：块 id 复用（GPU 池驱逐后重分配）会让缓存的前沿判定过期，
        故请求被抢占时前沿随 block_ids 一起重置；运行中请求的前缀块被
        ref 持有，不会被复用。
        """
        front = state.ledger_front[g]
        hit_at = state.ledger_hit_at[g]
        if hit_at >= 0 and hit_at <= pos:
            return True
        pool = self._gpu_block_pool
        if pool is None:
            return False
        ids = state.block_ids[g]
        blocks = pool.blocks
        while front <= pos and front < len(ids):
            blk = blocks[ids[front]]
            bhash = blk.block_hash
            if bhash is not None and profiler.was_ever_hit(bhash):
                state.ledger_hit_at[g] = front
                state.ledger_front[g] = front + 1
                return True
            front += 1
        state.ledger_front[g] = front
        return False

    def _write_gate_should_store(
        self,
        gpu_block: "KVCacheBlock",
        preempted: bool,
        ledger_hit: bool = False,
    ) -> bool:
        """WriteGate v3：逐块写准入。返回 False = 暂缓（不写盘，下步重扫）。

        暂缓永远安全：块未 offload 与被 gate 拒绝在未来未命中时的代价
        一致（重算），不会造成正确性问题。

        v1 教训：决策发生在块刚算完的时刻，此刻它只被当前请求持有
        （ref_cnt==1），而未来会被复用的前缀此刻恰恰是 ref_cnt==1——
        "当下共享"与"未来复用"负相关，一次定终身的丢弃把热点全丢了。
        v2 修法：拒绝不推进游标，后续步重扫；兄弟请求 lookup 命中后
        ref_cnt>1 即翻盘写盘。
        v3 新增：ledger（历史复用账本）与 auto（在线死写率闭环调档）。

        信号语义（逗号分隔，静态信号取并集）：
        - share：前缀共享度。ref_cnt>1（≥2 个请求同持 = 前缀主干）才写；
          ref_cnt==1（单请求私有尾缀）暂缓。依据：本池 ref_cnt 只由
          请求分配与前缀命中递增，radix 注册哈希不持引用（block_pool
          ._insert_block_hash），故该判据即"当下是否被共享"。
        - ledger：该块或其祖先曾从 offload 读回过（历史复用证据）才写。
        - lifecycle：被抢占请求的块无条件写（恢复时省重算）。
        - auto：由 WriteGateController 的档位决定——relax 全写 / mid
          share∨ledger / strict 仅 ledger；lifecycle 保底不过控制器。
        """
        signals = self._write_gate_signals
        if not signals:
            return True
        if "lifecycle" in signals and preempted:
            return True
        if "auto" in signals:
            tier = self._gate_ctrl.tier if self._gate_ctrl else GATE_RELAX
            if tier == GATE_RELAX:
                return True
            if tier == GATE_MID:
                return gpu_block.ref_cnt > 1 or ledger_hit
            return ledger_hit
        if "share" in signals and gpu_block.ref_cnt > 1:
            return True
        if "ledger" in signals and ledger_hit:
            return True
        if signals & {"share", "ledger"}:
            return False
        return True

    def prepare_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Prepare store specs for the store event."""
        if self._lazy_mode:
            return self._prepare_lazy_store_specs()
        else:
            return self._prepare_eager_store_specs(scheduler_output)

    def _prepare_lazy_store_specs(
        self,
    ) -> tuple[list[int], list[int], list[str]]:
        """Single-pass cursor walk: offload cached GPU blocks near eviction.

        Walks the GPU free queue from the cursor, counting blocks that are
        free-or-offloaded (safe for the allocator to evict). Stops when
        target_free blocks are covered or CPU capacity is reached.
        """
        gpu_pool = self._gpu_block_pool
        if gpu_pool is None or self._target_free <= 0:
            return [], [], []

        free_queue = gpu_pool.free_block_queue
        cpu_pool = self.cpu_block_pool
        num_cpu_free = cpu_pool.get_num_free_blocks()

        # Validate cursor: stale if block was removed from free queue.
        if self._cursor is not None and self._cursor.ref_cnt > 0:
            self._cursor = None

        gpu_ids: list[int] = []
        block_hashes: list[bytes] = []
        cpu_blocks: list[KVCacheBlock] = []
        last_visited = self._cursor
        seg_alloc = self._disk_seg_alloc

        for covered, node in enumerate(free_queue.iter_blocks_after(self._cursor)):
            if covered >= self._target_free or len(gpu_ids) >= num_cpu_free:
                break

            last_visited = node
            bhash = node.block_hash

            if (
                bhash is not None
                and not node.is_null
                and cpu_pool.cached_block_hash_to_block.get_one_block(bhash) is None
            ):
                if seg_alloc is not None:
                    # 阶段二段亲和：GPU 块 id 邻近（同请求/同前缀落同 GPU 段）
                    # 时聚到同磁盘段，保持 run 连续性；否则回退全局 bump
                    aff_key = f"g{node.block_id // 32}"
                    cpu_blk = seg_alloc.take_block_affinity(aff_key)
                    if cpu_blk is None:
                        break
                    cpu_blocks.append(cpu_blk)
                gpu_ids.append(node.block_id)
                block_hashes.append(bhash)

        self._cursor = last_visited

        # Batch-allocate CPU blocks and stamp hashes.
        if gpu_ids:
            if seg_alloc is None:
                cpu_blocks = cpu_pool.get_new_blocks(len(gpu_ids))
            cpu_ids = [blk.block_id for blk in cpu_blocks]
            for cpu_blk, bhash in zip(cpu_blocks, block_hashes):  # type: ignore[assignment]
                cpu_blk._block_hash = bhash  # type: ignore[assignment]
            # Touch GPU blocks to prevent eviction during async copy.
            gpu_pool.touch([gpu_pool.blocks[bid] for bid in gpu_ids])
        else:
            cpu_ids = []

        if profiler.PROFILE:
            # 体积账：写侧逐块决策。WriteGate 接入前 dropped 恒为 0。
            profiler.note_store_decision(len(gpu_ids), 0, block_hashes)

        return gpu_ids, cpu_ids, []

    def _prepare_eager_store_specs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[list[int], list[int], list[str]]:
        """Identify newly computed blocks to offload from scheduler requests.

        Only considers blocks whose KV data has been **confirmed computed** by
        the GPU. This means blocks from the current step are NOT stored until the
        next step. If a request finishes in the same step as its last full block,
        that block may be missed. (TODO: flush on finish.)

        Returns:
            (gpu_block_ids, cpu_block_ids, req_ids) for the store event.
        """

        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        merged_block_hashes: list[bytes] = []
        merged_dropped_blocks = 0
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        seg_alloc = self._disk_seg_alloc
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        # Dedup against blocks already scheduled.
        in_flight = self._in_flight_store_gpu_blocks

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            # Accumulate new block IDs.
            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
                # 旧块 id 随抢占失效，重试队列与 ledger 前沿一并重置
                state.gate_pending.clear()
                state.ledger_front = [0] * num_groups
                state.ledger_hit_at = [-1] * num_groups
            if new_block_id_groups:
                for g in range(min(num_groups, len(new_block_id_groups))):
                    if new_block_id_groups[g] is not None:
                        state.block_ids[g].extend(new_block_id_groups[g])

            num_new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if num_new_tokens == 0:
                continue

            block_ids_by_group = state.block_ids
            if not block_ids_by_group:
                continue

            # --- Phase 1: Scan blocks, classify as cached vs to-store ---
            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            n_dropped_req = 0
            # 段式分配时逐块在此获取 CPU/磁盘块（bump 连续）；否则扫描后批量分配
            cpu_blocks_alloc: list[KVCacheBlock] = []
            advanced_per_group: list[int] = [0] * num_groups
            out_of_space = False
            # Confirmed tokens: KV data written and visible to all streams.
            req = state.request
            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            # Cap to blocks with confirmed KV data.
            aligned_tokens = confirmed_tokens // self.block_size * self.block_size

            # --- Phase 0: WriteGate 重试暂缓块（先于新块扫描） ---
            if state.gate_pending:
                still_pending: list[tuple[int, int, int]] = []
                for g, bid, pos in state.gate_pending:
                    blk = gpu_block_pool.blocks[bid]
                    if blk.is_null or blk.block_hash is None:
                        continue  # 块失效，放弃重试
                    if (
                        bid in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            blk.block_hash
                        )
                        is not None
                    ):
                        continue  # 已在写或 CPU 池已有同 hash
                    ledger_hit = (
                        self._gate_ledger and self._gate_family_hit(state, g, pos)
                    )
                    if self._write_gate_should_store(blk, preempted, ledger_hit):
                        if seg_alloc is not None:
                            aff_key = f"g{bid // 32}"
                            cpu_blk = seg_alloc.take_block_affinity(aff_key)
                            if cpu_blk is None:
                                out_of_space = True
                                still_pending.append((g, bid, pos))
                                continue
                            cpu_blocks_alloc.append(cpu_blk)
                        else:
                            if num_free <= 0:
                                out_of_space = True
                                still_pending.append((g, bid, pos))
                                continue
                            num_free -= 1
                        gpu_block_ids.append(bid)
                        block_hashes_to_store.append(blk.block_hash)
                    else:
                        still_pending.append((g, bid, pos))
                state.gate_pending = still_pending

            for g in range(num_groups):
                # FIXME (yifan): handle CPU cache eviction, where
                # num_stored_blocks can be stale and omit evicted blocks in
                # the middle of the request.
                already_stored_g = state.num_stored_blocks[g]
                group_gpu_ids = block_ids_by_group[g]

                g_block_size = (
                    kv_cache_groups[g].kv_cache_spec.block_size * self.cp_world_size
                )
                ready_blocks_g = aligned_tokens // g_block_size
                scannable = group_gpu_ids[already_stored_g:ready_blocks_g]

                for pos, gpu_block_id in enumerate(scannable, start=already_stored_g):
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[g] += 1
                        continue

                    bhash_with_group = gpu_block.block_hash
                    if bhash_with_group is None:
                        # Masked-out SWA position the coordinator chose not to
                        # hash; it can never serve a prefix-cache hit, so skip.
                        advanced_per_group[g] += 1
                        continue

                    # Skip if already scheduled for store or already cached in CPU.
                    if (
                        gpu_block_id in in_flight
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    # WriteGate v2/v3：逐块写准入。拒绝 = 暂缓入 per-request
                    # 重试队列（游标照常推进，位置 pos 一并入队供 ledger 祖先
                    # 查询）；每步扫描前重试 pending，兄弟请求 lookup 命中把
                    # ref_cnt 顶上去、或前缀家族出现在读回账本后翻盘写盘。
                    # 体积账按 hash 首丢去重（块 id 会被复用，不可作键）。
                    ledger_hit = (
                        self._gate_ledger and self._gate_family_hit(state, g, pos)
                    )
                    if not self._write_gate_should_store(
                        gpu_block, preempted, ledger_hit
                    ):
                        advanced_per_group[g] += 1
                        state.gate_pending.append((g, gpu_block_id, pos))
                        if bhash_with_group not in self._gate_dropped_hashes:
                            self._gate_dropped_hashes.add(bhash_with_group)
                            n_dropped_req += 1
                        continue

                    if seg_alloc is not None:
                        # 阶段二段亲和：同请求块聚同段（GPU 块 id 邻近为 key）
                        aff_key = f"g{gpu_block_id // 32}"
                        cpu_blk = seg_alloc.take_block_affinity(aff_key)
                        if cpu_blk is None:
                            out_of_space = True
                            break
                        cpu_blocks_alloc.append(cpu_blk)
                    else:
                        if num_free <= 0:
                            out_of_space = True
                            break
                        num_free -= 1

                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(bhash_with_group)
                    advanced_per_group[g] += 1

                if out_of_space:
                    break

            # --- Phase 2: Batch allocate CPU blocks and stamp hashes ---
            n_to_alloc = len(gpu_block_ids)
            if n_to_alloc > 0:
                if seg_alloc is None:
                    cpu_blocks_alloc = cpu_block_pool.get_new_blocks(n_to_alloc)
                cpu_block_ids = [blk.block_id for blk in cpu_blocks_alloc]
                for cpu_blk, bhash in zip(cpu_blocks_alloc, block_hashes_to_store):
                    cpu_blk._block_hash = bhash  # type: ignore[assignment]
            else:
                cpu_block_ids = []

            if cpu_block_ids:
                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                if profiler.PROFILE:
                    merged_block_hashes.extend(block_hashes_to_store)
                in_flight.update(gpu_block_ids)

                # Touch GPU blocks to prevent freeing during async copy
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[bid] for bid in gpu_block_ids]
                )

                logger.debug(
                    "Request %s: Scheduling store of %d blocks to CPU (%d groups)",
                    req_id,
                    len(cpu_block_ids),
                    num_groups,
                )

            # Advance per-group cursors (includes cached hits + newly stored)
            for g in range(num_groups):
                state.num_stored_blocks[g] += advanced_per_group[g]
            merged_dropped_blocks += n_dropped_req

        if profiler.PROFILE and (merged_gpu_block_ids or merged_dropped_blocks):
            # 体积账：写侧逐块决策（写盘 / WriteGate 拒绝）。
            profiler.note_store_decision(
                len(merged_gpu_block_ids), merged_dropped_blocks,
                merged_block_hashes,
            )
            if self._gate_ctrl is not None:
                # 闭环反馈回路：把本步写盘块推入存活窗，成熟后回算在线死写率，
                # 由死写率驱动准入档位；档位快照随 profiler 分片落盘。
                self._gate_ctrl.note_step(
                    merged_block_hashes, merged_dropped_blocks
                )
                profiler.note_gate_state(self._gate_ctrl.snapshot())

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        """Handle async transfer completions from worker.

        Load completions arrive via finished_recving (real req_ids).
        Store completions arrive via kv_connector_worker_meta as
        per-event worker counts. We accumulate across steps and process
        a store event only when all workers have reported completion.
        """
        # --- Load completions ---
        for req_id in list(connector_output.finished_recving or []):
            self._cleanup_load_request(req_id)

        # --- Store completions ---
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, SimpleCPUOffloadWorkerMetadata):
            return
        for event_idx, count in meta.completed_store_events.items():
            total = self._store_event_pending_counts.get(event_idx, 0) + count
            if total >= self._expected_worker_count:
                self._store_event_pending_counts.pop(event_idx, None)
                self._process_store_event(event_idx)
            else:
                self._store_event_pending_counts[event_idx] = total

    def _process_store_event(self, event_idx: int) -> None:
        """Process a fully-completed store event."""
        transfer = self._store_event_to_blocks.pop(event_idx, None)
        if transfer is None:
            transfer = self._abandoned_store_event_to_blocks.pop(event_idx, None)
            if transfer is None:
                return  # guard stale events from before a reset() call
            self._release_transfer_refs(transfer)
            return

        if not self._lazy_mode:
            self._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)

        self._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids)
        logger.debug(
            "Store event %d completed: cached %d blocks to CPU",
            event_idx,
            len(transfer.cpu_block_ids),
        )

        # Eager only: update per-req state
        if not self._lazy_mode:
            for req_id in self._store_event_to_reqs.pop(event_idx, []):
                state = self._reqs_to_store.get(req_id)
                if state is None:
                    continue
                state.store_events.discard(event_idx)
                if state.finished and not state.store_events:
                    self._cleanup_store_request(req_id)

    def _process_store_completion(
        self, gpu_block_ids: list[int], cpu_block_ids: list[int]
    ) -> None:
        """Cache CPU blocks per-group and release GPU refs.

        Block hashes were stamped on CPU blocks at allocation time (in
        ``_prepare_*_store_specs``).  Here we just register them in the
        cache map so they become discoverable by the load path.
        """
        assert len(cpu_block_ids) == len(gpu_block_ids)

        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in cpu_block_ids]

        for cpu_block in cpu_blocks:
            bhash = cpu_block.block_hash
            assert bhash is not None
            self.cpu_block_pool.cached_block_hash_to_block.insert(bhash, cpu_block)

        # Free CPU and GPU blocks' ref counts to turn them into prefix cache
        self._free_cpu_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in gpu_block_ids
        )

    def _release_transfer_refs(self, transfer: TransferMeta) -> None:
        """Release transfer refs without making copied data cacheable."""
        cpu_blocks = [self.cpu_block_pool.blocks[bid] for bid in transfer.cpu_block_ids]
        for cpu_block in cpu_blocks:
            cpu_block.reset_hash()
        self._free_cpu_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in transfer.gpu_block_ids
        )

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(
            self._store_event_to_blocks or self._abandoned_store_event_to_blocks
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Always returns (False, None). GPU blocks are protected by ref_cnt,
        so the scheduler can free blocks immediately."""
        req_id = request.request_id

        # Release any temp CPU hit pin from get_num_new_matched_tokens()
        # if request is canceled or preempted before update_state_after_alloc()
        pending = self._pending_cpu_hits.pop(req_id, None)
        if pending is not None:
            self._free_pending_cpu_hit(pending)

        # Handle load: defer cleanup if load is in-flight
        load_state = self._reqs_to_load.get(req_id)
        if load_state is not None:
            if load_state.load_event is not None:
                load_state.finished = True  # Defer: load in-flight
            else:
                self._cleanup_load_request(req_id)

        # Handle store (eager mode only): defer cleanup if stores in-flight
        if not self._lazy_mode:
            store_state = self._reqs_to_store.get(req_id)
            if store_state is not None:
                if store_state.store_events:
                    store_state.finished = True  # Defer: stores in-flight
                else:
                    self._cleanup_store_request(req_id)

        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.request_finished(request, block_ids=[])

    def _free_cpu_blocks(self, blocks: "Iterable[KVCacheBlock]") -> None:
        """释放 CPU/磁盘池块，并通知段分配器做整段回收检查。"""
        block_list = list(blocks)
        self.cpu_block_pool.free_blocks(block_list)
        if self._disk_seg_alloc is not None:
            self._disk_seg_alloc.note_freed(b.block_id for b in block_list)

    def _free_pending_cpu_hit(self, pending: tuple) -> None:
        """Release the temporary CPU block pin taken in get_num_new_matched_tokens()."""
        cpu_hit_blocks, _ = pending
        blocks_to_free = [
            blk for grp in cpu_hit_blocks for blk in grp if not blk.is_null
        ]
        if blocks_to_free:
            self._free_cpu_blocks(blocks_to_free)

    def _cleanup_load_request(self, req_id: str) -> None:
        """Release all load resources for a request.

        Shared between request_finished() and update_connector_output() paths.
        Removes the request from _reqs_to_load, cleans up event mappings,
        and frees CPU/GPU touch refs.
        """
        state = self._reqs_to_load.pop(req_id, None)
        if state is None:
            state = self._abandoned_reqs_to_load.pop(req_id, None)
        if state is None:
            return
        # Remove from load event mapping (only this req, not whole event)
        if state.load_event is not None:
            reqs = self._load_event_to_reqs.get(state.load_event)
            if reqs is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._load_event_to_reqs.pop(state.load_event, None)

        if state.transfer_meta is not None:
            # Free CPU touch refs
            self._free_cpu_blocks(
                self.cpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.cpu_block_ids
            )
            # Free GPU touch refs
            assert self._gpu_block_pool is not None
            self._gpu_block_pool.free_blocks(
                self._gpu_block_pool.blocks[bid]
                for bid in state.transfer_meta.gpu_block_ids
            )

    def _cleanup_store_request(self, req_id: str) -> None:
        """Release store metadata for a request.

        Metadata-only cleanup but no block freeing. Job completion handles
        block caching and GPU ref freeing via _process_store_completion().
        """
        state = self._reqs_to_store.pop(req_id, None)
        if state is None:
            return
        for event_idx in list(state.store_events):
            if (reqs := self._store_event_to_reqs.get(event_idx)) is not None:
                with contextlib.suppress(ValueError):
                    reqs.remove(req_id)
                if not reqs:
                    self._store_event_to_reqs.pop(event_idx, None)
        state.store_events.clear()

    def take_events(self) -> Iterable[KVCacheEvent]:
        return self.cpu_block_pool.take_events()

    def reset(self) -> bool:
        """Abandon pending transfers and reset the CPU cache when safe.

        Worker-side DMA may still be using blocks after reset is requested.
        Keep those block refs pinned until the existing completion path reports
        the transfer finished, then release refs without caching abandoned
        store results.
        """

        self._abandoned_store_event_to_blocks.update(self._store_event_to_blocks)
        self._store_event_to_blocks.clear()
        self._in_flight_store_gpu_blocks.clear()

        # Loads that have not been sent to the worker cannot have running DMA.
        # In-flight loads stay pinned and are cleaned up on completion.
        for req_id in list(self._reqs_to_load):
            state = self._reqs_to_load.pop(req_id)
            if state.load_event is None:
                self._reqs_to_load[req_id] = state
                self._cleanup_load_request(req_id)
            else:
                self._abandoned_reqs_to_load[req_id] = state

        self._reqs_to_store.clear()
        self._store_event_to_reqs.clear()
        self._store_event_pending_counts = {
            event_idx: count
            for event_idx, count in self._store_event_pending_counts.items()
            if event_idx in self._abandoned_store_event_to_blocks
        }
        self._cursor = None
        # NOTE: _load_event_counter / _store_event_counter are not
        # reset as they are monotonic and must stay ahead of the workers
        # high-water marks to avoid event index collisions

        if self._abandoned_store_event_to_blocks or self._abandoned_reqs_to_load:
            return False

        return self.cpu_block_pool.reset_prefix_cache()
