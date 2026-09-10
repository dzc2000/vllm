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
    # WriteGate v6：直供 load 对 (pending_slot, gpu_block) 与命中 pin 的
    # hash 列表（load 完成后释放 pin，见 _cleanup_load_request）。
    v6_pairs: list = field(default_factory=list)
    pending_pins: list = field(default_factory=list)


class _DeferPendingPool:
    """WriteGate v6（run6 defer）延迟写池：调度侧纯账本。

    数据本体在 worker 侧 pinned pending 缓冲（DiskBackend），本类只记
    hash -> 槽位与状态机。构造性不变量 INV1：落盘只经首读兑现
    （_v6_redeem），pending 未读块最坏路径 = 重算，永不落盘。

    状态机（per hash）：
      inflight（DMA 入池未完成，不可命中）
      -> settled（可命中）
      -> pinned（命中/落盘保护中，LRU 不可逐出）
      -> flushed（盘上有副本；条目保留为 RAM 缓存直至 LRU 逐出）
    LRU 逐出未 flush 条目 = 丢弃（数据消失，重算兜底，经 on_drop 计入
    体积账 dropped_blocks，unique-hash 去重）；丢无可丢（全 pinned/
    inflight）时拒绝新入池 = 最终丢弃。丢弃后的 hash 被重算（= 需求
    证据）可再次入池（二次机会，不重复计数）。

    S4b（§16）逐出写回：传入 on_evict_wb 后"未 flush 逐出"改走"先写回
    再让容量"——回调返回 True（已入队盘侧）时槽位被 pwritev 占用不可
    交还，本轮不产出空槽（admit 拒绝新 hash，RAM 驻留仍 <= cap，W1 等
    容量构造性成立）；flush 完成 hook 退针后该条目经"已 flush"路径无损
    让出容量。回调返回 False（盘池满）落回原 drop 路径。
    None = v6 原生 drop 语义（逐字节不变）。
    """

    def __init__(self, cap_slots: int, on_drop=None, on_evict_wb=None) -> None:
        self.cap = max(0, int(cap_slots))
        self._slot: dict[bytes, int] = {}  # hash -> pending 槽位
        self._lru: dict[bytes, None] = {}  # 插入序即 LRU 序（touch 重插）
        self._free: list[int] = list(range(self.cap))[::-1]
        self.inflight: set[bytes] = set()
        self.pinned: dict[bytes, int] = {}  # hash -> pin 计数
        self.flushed: set[bytes] = set()
        self.dropped: set[bytes] = set()
        self._on_drop = on_drop
        # S4b WB 逐出写回回调 (hash, pslot) -> bool，False = 写不回走原 drop
        self._on_evict_wb = on_evict_wb
        # 计数器（判卷护栏与分列字段的原始数据）
        self.admits = 0
        self.hits = 0
        self.drops = 0
        self.flushes = 0
        self.wb_flushes = 0  # flushes 中由逐出写回贡献的子集
        self.dropped_then_requested = 0
        self.peak = 0

    def contains(self, h: bytes) -> bool:
        return h in self._slot

    def touch(self, h: bytes) -> None:
        if h in self._lru:
            del self._lru[h]
            self._lru[h] = None  # 重插到尾 = MRU

    def _drop(self, h: bytes) -> None:
        if h not in self.dropped:
            self.dropped.add(h)
            self.drops += 1
            if self._on_drop is not None:
                self._on_drop(h)

    def _evict_one(self) -> bool:
        for h in self._lru:  # dict 序 = LRU 序，从最老开始
            if self.pinned.get(h, 0) > 0 or h in self.inflight:
                continue
            if h not in self.flushed and self._on_evict_wb is not None:
                # S4b WB：未落盘块先写回（回调置 flushed + pin_protect +
                # 盘侧排队）。成功时槽位被 pwritev 占用不可交还——本轮
                # 不产出空槽，返回 False 即 admit 拒绝新 hash；flush 完成
                # hook unpin 后，该条目经下方"已 flush"路径正常让出容量。
                if self._on_evict_wb(h, self._slot[h]):
                    return False
                # 盘池满：写回失败，落回原 drop 路径（v6 语义兜底）
            slot = self._slot.pop(h)
            del self._lru[h]
            self._free.append(slot)
            if h not in self.flushed:
                self._drop(h)  # 未 flush：数据消失，重算兜底
            # 已 flush：盘上有副本，逐出无损（hash 仍可从 CPU 池发现）
            return True
        return False  # 全被 pin/inflight：丢无可丢

    def admit(self, h: bytes) -> bool:
        """块入池（inflight 态）。满时先 LRU 丢最老未读；丢无可丢则拒绝。"""
        if h in self._slot:
            self.touch(h)
            return True
        if not self._free and not self._evict_one():
            self._drop(h)  # 池满拒绝 = 最终丢弃（重算兜底）
            return False
        slot = self._free.pop()
        self._slot[h] = slot
        self._lru[h] = None
        self.inflight.add(h)
        self.admits += 1
        self.peak = max(self.peak, len(self._slot))
        return True

    def settle(self, hashes: list[bytes]) -> None:
        """store 事件完成：入池 DMA 数据就绪，hash 变为可命中。"""
        for h in hashes:
            self.inflight.discard(h)

    def hittable(self, h: bytes) -> bool:
        return h in self._slot and h not in self.inflight

    def pin(self, h: bytes) -> "int | None":
        """命中：返回槽位并 pin（兑现完成前 LRU 不可逐出）。"""
        if not self.hittable(h):
            return None
        self.pinned[h] = self.pinned.get(h, 0) + 1
        self.hits += 1
        self.touch(h)
        return self._slot[h]

    def pin_protect(self, h: bytes) -> None:
        """落盘期间的槽位保护（非命中计数）：pwritev 未完成前 LRU 不可逐出。"""
        if h in self._slot:
            self.pinned[h] = self.pinned.get(h, 0) + 1

    def unpin(self, h: bytes) -> None:
        n = self.pinned.get(h, 0)
        if n <= 1:
            self.pinned.pop(h, None)
        else:
            self.pinned[h] = n - 1

    def mark_flushed(self, h: bytes) -> None:
        self.flushed.add(h)

    def slot_of(self, h: bytes) -> "int | None":
        return self._slot.get(h)

    def snapshot(self) -> dict:
        return {
            "enabled": self.cap > 0,
            "cap_slots": self.cap,
            "resident": len(self._slot),
            "inflight": len(self.inflight),
            "pinned": sum(1 for n in self.pinned.values() if n > 0),
            "admits": self.admits,
            "pending_hits": self.hits,
            "drops": self.drops,
            "flushes": self.flushes,
            "wb_flushes": self.wb_flushes,
            "dropped_set": len(self.dropped),
            "dropped_then_requested": self.dropped_then_requested,
            "pending_peak": self.peak,
        }


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
        self.num_segments_evicted: int = 0

    def take_block(self) -> "KVCacheBlock | None":
        """段内 bump 取下一个空闲 slot，并完成池记账（出队/逐 hash/引用计数）。

        容量耗尽（无整段空闲可激活）时返回 None，调用方按 out_of_space 处理。
        """
        pool = self._pool
        while True:
            if self._active < 0:
                if not self._free_segs and not self._evict_coldest_segment():
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
        """整段真空（每块 null，或 ref_cnt==0 且哈希已摘）时归还段空闲列表尾部。

        ref_cnt==0 但 block_hash 仍在的块是有效缓存块；提前回收其所在段
        等于逐出缓存，同一前缀再次落盘时只能重写（重写放大之源）。
        """
        if self._seg_free[seg]:
            return
        base = seg * self._seg_size
        end = min(base + self._seg_size, self._num_slots)
        if all(
            self._blocks[b].is_null
            or (
                self._blocks[b].ref_cnt == 0
                and self._blocks[b].block_hash is None
            )
            for b in range(base, end)
        ):
            self._seg_free[seg] = True
            self._free_segs.append(seg)
            self.num_segments_recycled += 1

    def _evict_coldest_segment(self) -> bool:
        """段耗尽时，按自由队列驱逐序逐出最冷缓存段并回收。

        从自由队列头（最冷）扫自由块，跳过仍被 pin（ref_cnt!=0）的段；
        命中含缓存块的段后摘掉段内全部缓存块哈希（数据不动、只删索引），
        再整段回收。找不到可逐出段（真容量压力）返回 False，调用方按
        out_of_space 处理。
        """
        pool = self._pool
        for blk in pool.free_block_queue.iter_blocks_after(None):
            if blk.is_null or blk.block_hash is None:
                continue
            seg = blk.block_id // self._seg_size
            base = seg * self._seg_size
            end = min(base + self._seg_size, self._num_slots)
            if any(
                not self._blocks[b].is_null and self._blocks[b].ref_cnt != 0
                for b in range(base, end)
            ):
                continue
            for b in range(base, end):
                cand = self._blocks[b]
                if not cand.is_null and cand.block_hash is not None:
                    pool._maybe_evict_cached_block(cand)
            self._try_recycle(seg)
            if self._seg_free[seg]:
                self.num_segments_evicted += 1
            return self._seg_free[seg]
        return False

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

# --- WriteGate v4（F1+F3）：开关、在线头保护窗 K、死头否决名单参数 -----------
# v4 把 auto 从"单一死写率驱动三档"改成"头/尾分窗 + 在线位置保护"：
#   F1 迟到命中复结算 —— 结算为死的块之后又被读回，则改判为活（在线口径
#      与论文体积账"任何时点命中即活"对齐，消掉 run3 实测 0.836 vs 0.562 的偏差）；
#   F3 头保护 + 在线 K —— 前缀头部 K 块无条件写（破 ledger 冷启动死锁），
#      K 由头窗死写率乘性调节，种子取 bud64 这个安全下界，让"手调 K"变成控制器不动点；
#   F3 死头否决 —— 越过 horizon 仍没人读回的头块进名单，只降级它的头通道特权
#      （证据通道照常），专治 bud512 式大 K 的 churn 放大（amp 3.637）。
# 默认关闭（VLLM_GATE_V4=1 打开），关闭时 v3 行为逐字节不变。
GATE_V4_DEFAULT = __import__("os").environ.get("VLLM_GATE_V4", "0") == "1"
GATE_K_SEED = 64  # 头保护窗初值（= run3 里唯一没判负的手调档）
GATE_K_MIN = 16
GATE_K_MAX = 4096
GATE_VETO_TTL_MULT = 8  # 否决有效期 = 8 x horizon，到期自动出名单
GATE_VETO_MAX = 131072  # 名单容量上限，超了按 FIFO 驱逐最老的

# --- WriteGate v5（run5 / F1-lite）：有界复结算，F3 零残留 ---------------------
# run4 尸检给出的分工：F1（迟到命中）是药——h1 上把读回抬到 29039、超过一切
# 静态点；F3（头保护 K）是毒——无条件写把 writes 撑爆 +16k~+28k、被 bud256
# 双端支配。v5 只留 F1 的信号源（读侧逻辑命中），改成更轻的有界形式：
#   一块没有账本证据、但读侧已出现过逻辑命中 -> 领一次重写特权。
#   领过的块进一次性集合永久除名；再死即永久死，回 v3 原证据通道。
#   不引入 K/否决/分窗——闭环仍是 v3 的三档，v5 只补它冷启动时看不见的那批块。
# 默认关闭（VLLM_GATE_V5=1 打开），关闭时 v3/v4 行为逐字节不变。
GATE_V5_DEFAULT = __import__("os").environ.get("VLLM_GATE_V5", "0") == "1"
GATE_V5_GRANT_MAX = 524288  # 一次性特权集合容量，超了按 FIFO 驱逐最老的


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
        v4=None,
        v5=None,
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
        # --- v4 状态（v4_enabled=False 时全部闲置，v3 路径零开销） -----------
        self.v4_enabled = GATE_V4_DEFAULT if v4 is None else bool(v4)
        self.protect_blocks = GATE_K_SEED  # 在线头保护窗 K
        # 分窗结算样本：[dead, hash]，用 list 而非 int 是为了能就地改判（F1）
        self._head_verdicts: deque[list] = deque(maxlen=window_blocks)
        self._tail_verdicts: deque[list] = deque(maxlen=window_blocks)
        # 写准入时登记的通道归属：hash -> "head"|"tail"，成熟时 pop 分窗
        self._kind: dict[bytes, str] = {}
        self._veto: dict[bytes, int] = {}  # hash -> 过期的 _seq 值
        self._veto_hits: dict[bytes, int] = {}  # 入名单时的命中数快照
        self._veto_order: deque[bytes] = deque()
        self._veto_ttl = GATE_VETO_TTL_MULT * horizon_blocks
        self._veto_max = GATE_VETO_MAX
        self.head_dead_rate = 0.0
        self.tail_dead_rate = 0.0
        self.k_grows = 0
        self.k_shrinks = 0
        self.revisits = 0  # F1：改判为活的块数
        self.veto_saved = 0  # 否决期出现迟到命中而翻案出名单
        self.veto_expired = 0  # TTL 到期自然出名单
        # --- v5 状态（v5_enabled=False 时全部闲置，v3/v4 路径零开销） --------
        self.v5_enabled = GATE_V5_DEFAULT if v5 is None else bool(v5)
        self._v5_granted: set[bytes] = set()  # 已消耗重写特权的块（一次性）
        self._v5_grant_order: deque[bytes] = deque()  # 触顶时 FIFO 驱逐顺序
        self._v5_grant_max = GATE_V5_GRANT_MAX
        self.v5_grants = 0  # 累计授予次数（每块至多一次，与集合大小同源）

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
            if self.v4_enabled:
                # v4：按写入时的通道归属分窗（头窗驱动 K，尾窗驱动档位）。
                # 没登记过的（gate 之外的写、或 _kind 兜底清空过）计入尾窗。
                kind = self._kind.pop(h, "tail")
                if kind == "head":
                    self._head_verdicts.append([dead, h])
                    if dead:
                        self._add_veto(h)
                else:
                    self._tail_verdicts.append([dead, h])

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
        if self.v4_enabled:
            self._rebalance_v4()
            return
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

    # --- v4：头/尾分窗闭环 ---------------------------------------------------

    def _snapshot_v4(self) -> dict:
        # v4 观测项：在线 K 及其动作、两个窗的死写率、F1 改判、否决名单流水
        return {
            "v4": True,
            "protect_blocks": self.protect_blocks,
            "k_grows": self.k_grows,
            "k_shrinks": self.k_shrinks,
            "head_dead_rate": round(self.head_dead_rate, 4),
            "tail_dead_rate": round(self.tail_dead_rate, 4),
            "head_window_size": len(self._head_verdicts),
            "revisits": self.revisits,
            "veto_size": len(self._veto),
            "veto_saved": self.veto_saved,
            "veto_expired": self.veto_expired,
        }

    def _add_veto(self, bhash) -> None:
        # 死头块进否决名单：记 TTL 与"入名单时的命中数"，之后命中数一旦增长
        # 就说明这块其实有用（迟到命中），当场翻案出名单。
        if bhash in self._veto:
            return
        while len(self._veto) >= self._veto_max and self._veto_order:
            old = self._veto_order.popleft()
            if self._veto.pop(old, None) is not None:
                self._veto_hits.pop(old, None)
        self._veto[bhash] = self._seq + self._veto_ttl
        self._veto_hits[bhash] = profiler.hit_counts([bhash])[0]
        self._veto_order.append(bhash)

    def _vetoed(self, bhash) -> bool:
        exp = self._veto.get(bhash)
        if exp is None:
            return False
        if self._seq >= exp:
            self._veto.pop(bhash, None)
            self._veto_hits.pop(bhash, None)
            self.veto_expired += 1
            return False
        if profiler.hit_counts([bhash])[0] > self._veto_hits.get(bhash, 0):
            self._veto.pop(bhash, None)
            self._veto_hits.pop(bhash, None)
            self.veto_saved += 1
            return False
        return True

    def decide_auto(self, gpu_block, ledger_hit: bool, bhash, pos: int) -> bool:
        # v4 的 auto 判据。头通道：pos 在保护窗内即无条件写（破 ledger 冷启动
        # 死锁），除非它已被否决（否决只剥夺头特权，证据通道照常放行）。
        # 被否决的头块必须继续走证据通道，因为它正是"头特权被证伪"的那批。
        head = bhash is not None and 0 <= pos < self.protect_blocks
        if head and self._vetoed(bhash):
            head = False
        if self.tier == GATE_RELAX:
            ok = True
        elif head:
            ok = True
        elif self.tier == GATE_MID:
            ok = gpu_block.ref_cnt > 1 or ledger_hit
        else:
            ok = ledger_hit
        if ok and bhash is not None:
            # 放行才登记：暂缓的块下一步重扫时还会再来，届时再登记。
            # 兜底：被 out_of_space 打断等原因残留的条目不该无限增长，宁可丢标签
            # （丢了下一次按 tail 统计，只是少一份头窗证据，不影响正确性）。
            if len(self._kind) > 262144:
                self._kind.clear()
            self._kind[bhash] = "head" if head else "tail"
        return ok

    def _resettle(self, window) -> None:
        # F1：结算为死、之后又被读回 -> 就地改判为活（与论文口径一致）。
        for v in window:
            if v[0] and profiler.was_ever_hit(v[1]):
                v[0] = 0
                self.revisits += 1

    def _rebalance_v4(self) -> None:
        self._resettle(self._tail_verdicts)
        self._resettle(self._head_verdicts)
        self._rebalance_tier_v4()
        self._rebalance_k_v4()

    def _rebalance_tier_v4(self) -> None:
        # 档位只由**尾窗**（纯证据通道）驱动，且 v4 去掉 v3 的 mid->relax 回退：
        # mid 期的低死写率本来就是"只写有证据的块"造出来的，拿它当放松回去的理由
        # 是自证伪（棘轮只收紧）。回退仍保留给冻结通道（strict->mid）。
        if len(self._tail_verdicts) < self.eval_every:
            return
        rate = sum(v[0] for v in self._tail_verdicts) / len(self._tail_verdicts)
        self.dead_rate = rate
        self.tail_dead_rate = rate
        if self.tier == GATE_RELAX:
            if rate >= self.UP_TO_MID:
                self._move(GATE_MID, rate, "dead_rate_v4")
        elif self.tier == GATE_MID:
            if rate >= self.UP_TO_STRICT:
                self._move(GATE_STRICT, rate, "dead_rate_v4")

    def _rebalance_k_v4(self) -> None:
        # 头保护窗 K 只由**头窗**驱动，乘性收缩/翻倍（界内钳位）。头窗样本攒得比
        # 尾窗慢（头块只占前缀的一小段），所以这里的 eval_every 复用同一节流值。
        # 迟滞带沿用 UP_TO_MID / DOWN_TO_RELAX，带内不动，避免 K 来回摆。
        if len(self._head_verdicts) < self.eval_every:
            return
        rate = sum(v[0] for v in self._head_verdicts) / len(self._head_verdicts)
        self.head_dead_rate = rate
        if rate >= self.UP_TO_MID:
            shrink = max(GATE_K_MIN, self.protect_blocks // 2)
            self._k_move(min(self.protect_blocks, shrink), rate, "head_dead")
        elif rate <= self.DOWN_TO_RELAX:
            grow = min(GATE_K_MAX, self.protect_blocks * 2)
            self._k_move(max(self.protect_blocks, grow), rate, "head_alive")

    def _k_move(self, target: int, rate: float, reason: str) -> None:
        prev = self.protect_blocks
        if target == prev:
            return  # 已顶到界，不记动作
        if target < prev:
            self.k_shrinks += 1
        else:
            self.k_grows += 1
        self.protect_blocks = target
        logger.info(
            "WriteGate v4: 头保护窗 K %d -> %d (头窗死写率 %.3f, 成熟 %d 块, %s)",
            prev,
            target,
            rate,
            len(self._head_verdicts),
            reason,
        )

    # --- v5（F1-lite）：有界重写特权 ------------------------------------------

    def grant_rewrite(self, bhash, ledger_hit: bool) -> bool:
        """F1-lite 判据：迟到命中的翻案，但每块的重写特权只发一次。

        授予条件（四连，缺一不领）：
          1. bhash 在场且此前没领过（一次性是本机制的全部 bounded 语义）；
          2. 没有账本证据（有 ledger 的块本来就会从 v3 原通道放行，
             不该消耗特权、也不该被记进一次性集合）；
          3. 读侧出现过逻辑命中 was_ever_hit（独立于写盘，破冷启动死锁
             的证据源：strict 下"没写过->没读回->永远没账本"的死循环，
             唯一能证明这块有用的信号只剩逻辑命中）。
        领过之后块若再次死掉，本判据恒 False——回落到 v3 的三档判断，
        该拒就拒。不做翻案回收（v4 的 _resettle/veto 那套这里全没有）。
        """
        if bhash is None or ledger_hit:
            return False
        if bhash in self._v5_granted:
            return False
        if not profiler.was_ever_hit(bhash):
            return False
        while len(self._v5_granted) >= self._v5_grant_max and self._v5_grant_order:
            # 触顶按 FIFO 驱逐最老的，而不是 clear：clear 会把老块的特权
            # 重新发出去，一次性语义就没了。run4 量级 unique ~30k，远不到顶。
            old = self._v5_grant_order.popleft()
            self._v5_granted.discard(old)
        self._v5_granted.add(bhash)
        self._v5_grant_order.append(bhash)
        self.v5_grants += 1
        return True

    def _snapshot_v5(self) -> dict:
        # v5 观测项：授予次数与一次性集合规模（判卷只读 summary，这些
        # 键进 pid 分片，供尸检核对"翻转量是否补上 v3 缺的 1.3~4.4%"）
        return {
            "v5": True,
            "v5_grants": self.v5_grants,
            "v5_granted_size": len(self._v5_granted),
        }

    # --- 可观测性 -----------------------------------------------------------

    def snapshot(self) -> dict:
        out = {
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
        if self.v4_enabled:
            out.update(self._snapshot_v4())
        if self.v5_enabled:
            out.update(self._snapshot_v5())
        return out


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
        hicache_min_hits: int = 1,
        trt_keep_head_blocks: int = 0,
        defer_pending_bytes: int = 0,
        evict_writeback: bool = False,
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
        # 语义基线参数：hicache = 块自身历史读回数 >= 阈值才写（SGLang
        # HiCache selective write-back 口径，按块不查祖先）；trtprio = 块在
        # 前缀中的位置落在 head 窗内才写（TRT-LLM 静态优先级/头前缀保留的
        # 近似，K=0 时该信号等价全拒，需配 share/ledger 之一并集使用）。
        self._hicache_min_hits = max(1, int(hicache_min_hits))
        self._trt_keep_head_blocks = max(0, int(trt_keep_head_blocks))
        # --- WriteGate v6（run6 defer）：延迟写 + 首读兑现 -------------------
        # defer_pending_bytes > 0 时激活（经 extra_config 的 defer_pending_gib
        # 换算，环境变量在 EngineCore 子进程不可靠）。仅 disk+eager 生效；
        # 激活后 eager 扫描的块全部改道 pending 池，逐块 gate 被旁路，
        # 落盘只经首读兑现（INV1）。与 v3/v4/v5 信号正交、与 lag 互斥。
        self._v6_pool: _DeferPendingPool | None = None
        self._v6_store_events: dict[int, tuple[list[int], list[bytes]]] = {}
        self._v6_flush_events: dict[int, list[tuple[bytes, int]]] = {}
        self._v6_hit_pins: dict[str, tuple[list[bytes], list[int]]] = {}
        self._v6_store_outbox: list[tuple[int, int]] = []
        self._v6_store_event_rec: tuple[list[int], list[bytes]] = ([], [])
        self._v6_flush_outbox: list[tuple[int, int]] = []
        self._v6_flush_event_rec: list[tuple[bytes, int]] = []
        if defer_pending_bytes > 0:
            if disk_capacity_bytes <= 0:
                logger.warning(
                    "WriteGate v6 defer 仅支持 disk 模式，defer_pending_bytes"
                    "=%d 被忽略", defer_pending_bytes)
            elif lazy_offload:
                logger.warning(
                    "WriteGate v6 defer 仅支持 eager 路径，defer_pending_bytes"
                    "=%d 被忽略", defer_pending_bytes)
            elif self.hash_block_size != self.fa_block_size:
                raise ValueError(
                    "WriteGate v6 defer 要求 hash_block_size == fa_block_size"
                    f"（{self.hash_block_size} != {self.fa_block_size}）")
            else:
                offload_capacity = (
                    disk_capacity_bytes if disk_capacity_bytes > 0
                    else cpu_capacity_bytes
                )
                # 反推块字节：num_cpu_blocks = floor(n_gpu*C//G) <= floor(C//B)
                # => C//num_cpu_blocks >= B，本侧槽位数 <= worker 侧缓冲槽数
                # （只会少用不会越界，方向安全；实际 2 的幂配置下两者相等）。
                block_bytes = max(
                    1, offload_capacity // max(1, self.num_cpu_blocks))
                slots = defer_pending_bytes // block_bytes
                self._v6_pool = _DeferPendingPool(
                    slots,
                    on_drop=(lambda h: profiler.note_store_decision(0, 1, []))
                    if profiler.PROFILE else None,
                    on_evict_wb=(self._v6_evict_writeback
                                 if evict_writeback else None),
                )
                logger.info(
                    "SimpleCPUOffloadScheduler: WriteGate v6 defer 池 %d 槽"
                    "（%.2f GiB，block=%d B，evict=%s）—— "
                    "落盘仅经首读兑现（INV1）",
                    slots, defer_pending_bytes / (1024**3), block_bytes,
                    "WB" if evict_writeback else "drop")
        elif evict_writeback:
            logger.warning(
                "WriteGate v6 evict_writeback 需要 defer 池"
                "（defer_pending_bytes>0），已忽略")
        # WriteGate v3：ledger（历史复用账本）与 auto（闭环控制器）都以
        # profiler 的读回命中表为唯一数据源。profiler 未开时该表恒空，
        # "无数据"会被误判成"无命中"而把写全部拒掉，故此处显式降级并告警。
        self._gate_ledger = bool(self._write_gate_signals & {"ledger", "auto"})
        self._gate_ctrl: WriteGateController | None = None
        # hicache 与 ledger/auto 同源（profiler 读回命中表）：未开 profile
        # 时 hit_counts 恒 0，会把写全拒掉，故一并显式降级。
        if (
            self._write_gate_signals & {"ledger", "auto", "hicache"}
            and not profiler.PROFILE
        ):
            logger.warning(
                "WriteGate signals %s 依赖 kvlog_profile 在线账本，但 profiler "
                "未激活 -> 丢弃 ledger/auto/hicache 信号，退化为 %s",
                sorted(self._write_gate_signals),
                sorted(self._write_gate_signals - {"ledger", "auto", "hicache"})
                or "关闭",
            )
            self._write_gate_signals -= {"ledger", "auto", "hicache"}
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
        # 重写去重：哈希口径在途集。块 id 去重挡不住"GPU 块逐出后同内容落新块"，
        # 而哈希要到 store 完成才注册进索引，期间同哈希会被重复判写。
        self._in_flight_store_hashes: dict[bytes, int] = {}
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
        # v6：上一轮调度遗留的命中 pin 释放（同请求重试/取消路径）
        if (stale_v6 := self._v6_hit_pins.pop(request.request_id, None)) \
                is not None and self._v6_pool is not None:
            for h in stale_v6[0]:
                self._v6_pool.unpin(h)

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

        if self._v6_pool is not None:
            served = self._v6_try_serve_from_pending(
                request.request_id, remaining_hashes, max_hit_len,
                cpu_hit_blocks, hit_length)
            if served is not None:
                return served, True

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
            v6_pin = self._v6_hit_pins.pop(req_id, None)
            if v6_pin is not None and self._v6_pool is not None:
                for h in v6_pin[0]:
                    self._v6_pool.unpin(h)
            return

        if pending is None:
            # --- v6：纯 pending 命中的兑现（内存直供 + 首读落盘） -----------
            v6_pin = self._v6_hit_pins.pop(req_id, None)
            if v6_pin is not None:
                self._v6_redeem(
                    req_id, request, block_ids_by_group, blocks,
                    num_external_tokens, v6_pin)
                return
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

    def _v6_try_serve_from_pending(
        self,
        req_id: str,
        remaining_hashes: list,
        max_hit_len: int,
        cpu_hit_blocks: tuple,
        cpu_hit_len: int,
    ) -> "int | None":
        """v6 命中裁决（run6 预登记读口径：pending 直供计入 reads）。

        裁决 1：盘命中的块整段仍驻 pending RAM -> 全量直供（免盘读，
        复读也走内存）；裁决 2：无盘命中 -> 从头走 pending 连续 run。
        mixed（盘命中 + pending 尾缀）不扩展：尾部重算、不落盘
        （未读不写，INV1 不破坏），是本实现的已知边界。
        """
        pool = self._v6_pool
        assert pool is not None
        hbs = self.hash_block_size
        n_max = max_hit_len // hbs
        if n_max <= 0:
            return None
        if cpu_hit_len > 0:
            hashes = [
                blk.block_hash
                for grp in cpu_hit_blocks
                for blk in grp
                if not blk.is_null and blk.block_hash is not None
            ]
            n_cpu = cpu_hit_len // hbs
            if n_cpu <= 0 or len(hashes) != n_cpu:
                return None
            if not all(pool.hittable(h) for h in hashes):
                return None
            slots: list[int] = []
            for h in hashes:
                s = pool.pin(h)
                if s is None:
                    for hh in hashes[:len(slots)]:
                        pool.unpin(hh)
                    return None
                slots.append(s)
            if profiler.PROFILE:
                profiler.note_block_loads(hashes)
            self._v6_hit_pins[req_id] = (hashes, slots)
            return n_cpu * hbs
        from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
        keys = [
            make_block_hash_with_group_id(h, self.fa_gidx)
            for h in remaining_hashes[:n_max]
        ]
        n_ext = 0
        for k in keys:
            if not pool.hittable(k):
                if k in pool.dropped:
                    pool.dropped_then_requested += 1
                break
            n_ext += 1
        if n_ext == 0:
            return None
        hashes = keys[:n_ext]
        slots = []
        for h in hashes:
            s = pool.pin(h)
            if s is None:
                for hh in hashes[:len(slots)]:
                    pool.unpin(hh)
                return None
            slots.append(s)
        if profiler.PROFILE:
            profiler.note_block_loads(hashes)
        self._v6_hit_pins[req_id] = (hashes, slots)
        return n_ext * hbs

    def _v6_redeem(
        self,
        req_id: str,
        request: "Request",
        block_ids_by_group,
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
        v6_pin: tuple,
    ) -> None:
        """v6 兑现：pending 内存直供 + 首读触发的落盘（写侧唯一出口）。

        INV1 构造点：flush 只在此发生，而此处每个 hash 都刚被 pin 过
        （>=1 次读），故落盘块死写率恒 0——判卷可直接核验
        volume.dead_blocks == 0。盘池满时本次不落盘（驻留 RAM，
        下次命中重试），INV1 不破坏。
        """
        pool = self._v6_pool
        assert pool is not None
        hashes, slots = v6_pin
        n_take = num_external_tokens // self.hash_block_size
        assert n_take > 0
        for h in hashes[n_take:]:  # 调度器少收的尾部退 pin
            pool.unpin(h)
        hashes = list(hashes[:n_take])
        slots = list(slots[:n_take])

        g = self.fa_gidx
        g_block_size = (
            self.cpu_kv_cache_config.kv_cache_groups[g].kv_cache_spec.block_size
            * self.cp_world_size
        )
        assert num_external_tokens % g_block_size == 0
        n_ext_g = num_external_tokens // g_block_size
        num_cached_fa_blocks = sum(
            blk.block_hash is not None for blk in blocks.blocks[self.fa_gidx]
        )
        num_computed_tokens = num_cached_fa_blocks * self.fa_block_size
        total_computed_tokens = num_computed_tokens + num_external_tokens
        n_computed_g = cdiv(total_computed_tokens, g_block_size)
        gpu_ext_start = n_computed_g - n_ext_g
        group_gpu_ids = block_ids_by_group[g]

        load_pairs: list[tuple[int, int]] = []
        flushed_now: list[bytes] = []
        seg_alloc = self._disk_seg_alloc
        for i, (h, pslot) in enumerate(zip(hashes, slots)):
            gpu_id = group_gpu_ids[gpu_ext_start + i]
            load_pairs.append((pslot, gpu_id))
            if h in pool.flushed:
                continue
            cpu_blk = None
            if seg_alloc is not None:
                aff_key = f"g{gpu_id // 32}"
                cpu_blk = seg_alloc.take_block_affinity(aff_key)
            elif self.cpu_block_pool.get_num_free_blocks() > 0:
                cpu_blk = self.cpu_block_pool.get_new_blocks(1)[0]
            if cpu_blk is None:
                continue  # 盘池满：不落盘，驻留 RAM 待下次命中重试
            cpu_blk._block_hash = h  # type: ignore[assignment]
            pool.mark_flushed(h)
            pool.pin_protect(h)  # pwritev 完成前 LRU 不可逐出该槽
            pool.flushes += 1
            flushed_now.append(h)
            self._v6_flush_outbox.append((pslot, cpu_blk.block_id))
            self._v6_flush_event_rec.append((h, cpu_blk.block_id))

        # GPU 块在异步直供 DMA 期间防复用（与原生 load 路径同式 touch）
        assert self._gpu_block_pool is not None
        gpu_ids = [gid for _, gid in load_pairs]
        self._gpu_block_pool.touch(
            [self._gpu_block_pool.blocks[bid] for bid in gpu_ids]
        )
        if profiler.PROFILE and flushed_now:
            # 体积账：兑现写 = 真实落盘（stored_blocks 只在此增长）；
            # 死写核验：flush 时该 hash 已有 >=1 次读（pin 时已记账）。
            profiler.note_store_decision(len(flushed_now), 0, flushed_now)
        assert self._reqs_to_load.get(req_id) is None
        self._reqs_to_load[req_id] = LoadRequestState(
            request=request,
            transfer_meta=TransferMeta(gpu_ids, []),
            v6_pairs=load_pairs,
            pending_pins=list(hashes),
        )

    def _v6_evict_writeback(self, h: bytes, pslot: int) -> bool:
        """S4b（§16）逐出写回：LRU 淘汰的未读块先落盘而非丢弃。

        复用 redeem 的盘侧入队式（无 GPU 参与）：seg_alloc 优先（"wb"
        亲和键，无历史亲和即回落全局段推进），无盘段分配器时走
        cpu_block_pool。成功后条目转 flushed 且驻留 RAM，仍可命中直读；
        同时计 flushes 与 wb_flushes（后者为 W2/W4 判据口径）。该 hash
        若日后被读回则非死写，never-read = 死写，正是 S4b 要测的量。
        盘池满返回 False，调用方落回 v6 drop 语义。
        """
        pool = self._v6_pool
        assert pool is not None
        seg_alloc = self._disk_seg_alloc
        cpu_blk = None
        if seg_alloc is not None:
            cpu_blk = seg_alloc.take_block_affinity("wb")
        elif self.cpu_block_pool.get_num_free_blocks() > 0:
            cpu_blk = self.cpu_block_pool.get_new_blocks(1)[0]
        if cpu_blk is None:
            return False  # 盘池满：写不回，drop 兜底
        cpu_blk._block_hash = h  # type: ignore[assignment]
        pool.mark_flushed(h)
        pool.pin_protect(h)  # pwritev 完成前 LRU 不可逐出该槽
        pool.flushes += 1
        pool.wb_flushes += 1
        self._v6_flush_outbox.append((pslot, cpu_blk.block_id))
        self._v6_flush_event_rec.append((h, cpu_blk.block_id))
        if profiler.PROFILE:
            profiler.note_store_decision(1, 0, [h])
        return True

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> SimpleCPUOffloadMetadata:
        # --- Stores ---
        store_event = -1
        store_gpu, store_cpu, store_req_ids = self.prepare_store_specs(scheduler_output)
        v6_store_pairs = self._v6_store_outbox
        v6_flush_pairs = self._v6_flush_outbox
        if store_gpu or v6_store_pairs or v6_flush_pairs:
            store_event = self._store_event_counter
            self._store_event_counter += 1
            self._store_event_to_blocks[store_event] = TransferMeta(
                store_gpu, store_cpu
            )
            if v6_store_pairs:
                self._v6_store_events[store_event] = self._v6_store_event_rec
            if v6_flush_pairs:
                self._v6_flush_events[store_event] = self._v6_flush_event_rec
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
        v6_load: list[tuple[int, int]] = []
        for req_id, load_state in self._reqs_to_load.items():
            if load_state.load_event is not None:
                continue
            assert load_state.transfer_meta is not None
            load_gpu.extend(load_state.transfer_meta.gpu_block_ids)
            load_cpu.extend(load_state.transfer_meta.cpu_block_ids)
            v6_load.extend(load_state.v6_pairs)
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
            v6_store_pairs=list(v6_store_pairs),
            v6_load_pairs=list(v6_load),
            v6_flush_pairs=list(v6_flush_pairs),
        )
        # v6 收件箱清空（本步已全部随 metadata 交给 worker）
        self._v6_store_outbox = []
        self._v6_store_event_rec = ([], [])
        self._v6_flush_outbox = []
        self._v6_flush_event_rec = []
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
        bhash: "bytes | None" = None,
        pos: int = -1,
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
        - hicache（基线）：该块**自身**历史读回数 >= hicache_min_hits 才写
          （HiCache selective 口径；与 ledger 的区别是不查祖先家族）。
        - trtprio（基线）：块在前缀中的位置 pos < trt_keep_head_blocks 才写
          （TRT-LLM 静态头前缀优先保留的近似；bhash/pos 缺一按不命中处理）。
        """
        signals = self._write_gate_signals
        if not signals:
            return True
        if "lifecycle" in signals and preempted:
            return True
        if "auto" in signals:
            ctrl = self._gate_ctrl
            if ctrl is not None and ctrl.v4_enabled:
                return ctrl.decide_auto(gpu_block, ledger_hit, bhash, pos)
            if ctrl is not None and ctrl.v5_enabled:
                # F1-lite 放在 v4 分派之后：v4 闭环有自己的通道登记，
                # 不与 v4 混线（run4 臂的复现因此完全不受影响）。
                if ctrl.grant_rewrite(bhash, ledger_hit):
                    return True
            tier = ctrl.tier if ctrl else GATE_RELAX
            if tier == GATE_RELAX:
                return True
            if tier == GATE_MID:
                return gpu_block.ref_cnt > 1 or ledger_hit
            return ledger_hit
        if "share" in signals and gpu_block.ref_cnt > 1:
            return True
        if "ledger" in signals and ledger_hit:
            return True
        if "trtprio" in signals and 0 <= pos < self._trt_keep_head_blocks:
            return True
        if "hicache" in signals and bhash is not None:
            if profiler.hit_counts([bhash])[0] >= self._hicache_min_hits:
                return True
        if signals & {"share", "ledger", "hicache", "trtprio"}:
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
                and bhash not in self._in_flight_store_hashes
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
            self._track_in_flight_hashes(block_hashes)
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
                        or blk.block_hash in self._in_flight_store_hashes
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            blk.block_hash
                        )
                        is not None
                    ):
                        continue  # 已在写或 CPU 池已有同 hash
                    ledger_hit = (
                        self._gate_ledger and self._gate_family_hit(state, g, pos)
                    )
                    if self._write_gate_should_store(
                        blk, preempted, ledger_hit, blk.block_hash, pos
                    ):
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
                        or bhash_with_group in self._in_flight_store_hashes
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            bhash_with_group
                        )
                        is not None
                    ):
                        advanced_per_group[g] += 1
                        continue

                    # --- v6 defer：逐块 gate 旁路，块入 pending 池（不落盘）。
                    # 落盘只经首读兑现（_v6_redeem 的 flush），INV1 构造成立。
                    # 池满拒绝的丢弃计数走 pool.on_drop 回调（体积账统一口径）。
                    if self._v6_pool is not None:
                        _pool = self._v6_pool
                        if _pool.contains(bhash_with_group):
                            _pool.touch(bhash_with_group)
                            advanced_per_group[g] += 1
                            continue
                        if _pool.admit(bhash_with_group):
                            self._v6_store_outbox.append(
                                (gpu_block_id, _pool.slot_of(bhash_with_group)))
                            self._v6_store_event_rec[0].append(gpu_block_id)
                            self._v6_store_event_rec[1].append(bhash_with_group)
                            gpu_block_pool.touch([gpu_block])
                            advanced_per_group[g] += 1
                            continue
                        # 池满丢无可丢 = 最终丢弃（重算兜底）
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
                        gpu_block, preempted, ledger_hit,
                        bhash_with_group, pos,
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
                self._track_in_flight_hashes(block_hashes_to_store)

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

        if profiler.PROFILE and (
            merged_gpu_block_ids or merged_dropped_blocks or self._v6_pool
        ):
            # 体积账：写侧逐块决策（写盘 / WriteGate 拒绝）。
            profiler.note_store_decision(
                len(merged_gpu_block_ids), merged_dropped_blocks,
                merged_block_hashes,
            )
            if self._v6_pool is not None:
                profiler.note_v6_state(self._v6_pool.snapshot())
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
        # --- v6：pending 入池 DMA 完成（settle）/ 兑现落盘完成（可发现） ---
        v6_rec = self._v6_store_events.pop(event_idx, None)
        if v6_rec is not None:
            gpu_ids, admit_hashes = v6_rec
            if self._v6_pool is not None:
                self._v6_pool.settle(admit_hashes)
            if gpu_ids:
                assert self._gpu_block_pool is not None
                self._gpu_block_pool.free_blocks(
                    self._gpu_block_pool.blocks[bid] for bid in gpu_ids
                )
        v6_flush = self._v6_flush_events.pop(event_idx, None)
        if v6_flush is not None:
            for h, cpu_bid in v6_flush:
                blk = self.cpu_block_pool.blocks[cpu_bid]
                bhash = blk.block_hash
                if bhash is not None:
                    # 落盘数据就绪，hash 变为可发现（与原生 store 完成同式）
                    self.cpu_block_pool.cached_block_hash_to_block.insert(
                        bhash, blk)
                if self._v6_pool is not None:
                    self._v6_pool.unpin(h)  # 释放落盘保护 pin
                self._free_cpu_blocks([blk])
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
            self._untrack_in_flight_hash(bhash)
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
            if cpu_block.block_hash is not None:
                self._untrack_in_flight_hash(cpu_block.block_hash)
            cpu_block.reset_hash()
        self._free_cpu_blocks(cpu_blocks)
        assert self._gpu_block_pool is not None
        self._gpu_block_pool.free_blocks(
            self._gpu_block_pool.blocks[bid] for bid in transfer.gpu_block_ids
        )

    def _track_in_flight_hashes(self, hashes: Iterable[bytes]) -> None:
        for h in hashes:
            self._in_flight_store_hashes[h] = (
                self._in_flight_store_hashes.get(h, 0) + 1
            )

    def _untrack_in_flight_hash(self, bhash: bytes) -> None:
        n = self._in_flight_store_hashes.get(bhash, 0)
        if n <= 1:
            self._in_flight_store_hashes.pop(bhash, None)
        else:
            self._in_flight_store_hashes[bhash] = n - 1

    def has_pending_stores(self) -> bool:
        """Return True if there are in-flight store transfers."""
        return bool(
            self._store_event_to_blocks
            or self._abandoned_store_event_to_blocks
            or self._v6_store_events
            or self._v6_flush_events
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
        # v6：未兑现的命中 pin 释放（请求取消/抢占路径）
        v6_pin = self._v6_hit_pins.pop(req_id, None)
        if v6_pin is not None and self._v6_pool is not None:
            for h in v6_pin[0]:
                self._v6_pool.unpin(h)

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
        # v6：load 完成（或废弃），pending 命中 pin 释放（槽位回归 LRU）
        if state.pending_pins and self._v6_pool is not None:
            for h in state.pending_pins:
                self._v6_pool.unpin(h)
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
        self._in_flight_store_hashes.clear()

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
