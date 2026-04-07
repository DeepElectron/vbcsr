#ifndef VBCSR_DETAIL_VBCSR_BACKEND_HANDLE_HPP
#define VBCSR_DETAIL_VBCSR_BACKEND_HANDLE_HPP

#include "backend_handle_common.hpp"

namespace vbcsr::detail {

template <typename T, typename Kernel>
struct VBCSRMatrixBackend {
    using Storage = ShapeBlockStore<T>;
    // The outer VBCSR backend still uses "slot" for the flat graph nonzero-block
    // index. The storage layer below is block-oriented and distinguishes matrix
    // block indices, shape block indices, and page block indices.

    static constexpr uint32_t hard_safe_slots_per_page() {
        return Storage::hard_safe_blocks_per_page();
    }

    static uint32_t normalize_configured_max_slots_per_page(uint32_t requested) {
        if (requested == 0) {
            return hard_safe_slots_per_page();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(
                requested,
                1u,
                static_cast<uint64_t>(hard_safe_slots_per_page())));
    }

    enum class ApplyMode {
        Scalar,
        Batched
    };

    struct ShapeApplyStats {
        std::atomic<uint64_t> scalar_batches{0};
        std::atomic<uint64_t> scalar_blocks{0};
        std::atomic<uint64_t> batched_batches{0};
        std::atomic<uint64_t> batched_blocks{0};
    };

    struct SpMMStats {
        int row_dim = 0;
        int inner_dim = 0;
        int col_dim = 0;
        std::atomic<uint64_t> launched_batches{0};
        std::atomic<uint64_t> launched_products{0};
    };

    struct ShapeBatchView {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        int page_id = -1;
        typename Storage::ShapePage page{};
        const ShapeApplyStats* stats = nullptr;

        uint32_t block_count() const {
            return page.block_count;
        }

        uint32_t block_capacity() const {
            return page.blocks_per_page;
        }

        int logical_slot(uint32_t block_index) const {
            return page.matrix_block(block_index);
        }

        const T* block_ptr(uint32_t block_index) const {
            return page.block_ptr(block_index);
        }

        uint64_t scalar_apply_batch_count() const {
            return stats == nullptr
                ? 0
                : stats->scalar_batches.load(std::memory_order_relaxed);
        }

        uint64_t batched_apply_batch_count() const {
            return stats == nullptr
                ? 0
                : stats->batched_batches.load(std::memory_order_relaxed);
        }
    };

    struct VBCSRApplyPlan {
        std::vector<ShapeBatchView> batches;
    };

    struct SpMMPolicyKey {
        int row_dim = 0;
        int inner_dim = 0;
        int col_dim = 0;

        bool operator<(const SpMMPolicyKey& other) const {
            return std::tie(row_dim, inner_dim, col_dim) <
                   std::tie(other.row_dim, other.inner_dim, other.col_dim);
        }
    };

    VBCSRMatrixBackend() = default;

    explicit VBCSRMatrixBackend(uint32_t max_slots_per_page)
        : storage(max_slots_per_page) {}

    VBCSRMatrixBackend(const VBCSRMatrixBackend&) = delete;
    VBCSRMatrixBackend& operator=(const VBCSRMatrixBackend&) = delete;

    VBCSRMatrixBackend(VBCSRMatrixBackend&& other) noexcept
        : blk_handles(std::move(other.blk_handles)),
          storage(std::move(other.storage)),
          contiguous_layout(other.contiguous_layout),
          shape_apply_stats(std::move(other.shape_apply_stats)),
          spmm_policy_lookup(std::move(other.spmm_policy_lookup)),
          spmm_stats_records(std::move(other.spmm_stats_records)) {}

    VBCSRMatrixBackend& operator=(VBCSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            blk_handles = std::move(other.blk_handles);
            storage = std::move(other.storage);
            contiguous_layout = other.contiguous_layout;
            shape_apply_stats = std::move(other.shape_apply_stats);
            {
                std::lock_guard<std::mutex> lock(apply_plan_mutex);
                apply_plan.reset();
            }
            std::lock_guard<std::mutex> lock(policy_mutex);
            spmm_policy_lookup = std::move(other.spmm_policy_lookup);
            spmm_stats_records = std::move(other.spmm_stats_records);
        }
        return *this;
    }

    // Per-logical-slot payload handle. The slot index is the flat local nonzero-block
    // index from row_ptr/col_ind; the handle locates that block's shape-page payload.
    std::vector<uint64_t> blk_handles;
    Storage storage;
    bool contiguous_layout = false;
    std::vector<std::unique_ptr<ShapeApplyStats>> shape_apply_stats;
    mutable std::mutex apply_plan_mutex;
    mutable std::unique_ptr<VBCSRApplyPlan> apply_plan;
    mutable std::mutex policy_mutex;
    mutable std::map<SpMMPolicyKey, size_t> spmm_policy_lookup;
    mutable std::vector<std::unique_ptr<SpMMStats>> spmm_stats_records;

    size_t local_scalar_nnz() const {
        return storage.scalar_value_count();
    }

    int shape_class_count() const {
        return storage.shape_count();
    }

    // This is the configured global cap forwarded to ShapeBlockStore. Each shape
    // can still realize a smaller page size, and existing shape pages keep their
    // current size until the storage is rebuilt.
    uint32_t configured_max_slots_per_page() const {
        return storage.max_blocks_per_page();
    }

    void set_configured_max_slots_per_page(uint32_t max_slots_per_page) {
        storage.set_max_blocks_per_page(max_slots_per_page);
        invalidate_apply_plan();
    }

    int ensure_shape(int row_dim, int col_dim, size_t expected_block_count = 0) {
        const int shape_id =
            storage.get_or_create_shape(row_dim, col_dim, expected_block_count);
        ensure_shape_stats(shape_id);
        invalidate_apply_plan();
        return shape_id;
    }

    uint64_t append_block_for_shape(int shape_id, int logical_slot) {
        const uint64_t handle = storage.append(shape_id, logical_slot);
        invalidate_apply_plan();
        return handle;
    }

    void rebuild_handle_table() {
        storage.rebuild_handles(blk_handles);
        invalidate_apply_plan();
    }

    const VBCSRApplyPlan& ensure_apply_plan() const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (!apply_plan) {
            auto plan = std::make_unique<VBCSRApplyPlan>();
            storage.for_each_page([&](const typename Storage::ShapePage& page) {
                plan->batches.push_back(ShapeBatchView{
                    page.shape_id,
                    page.row_dim,
                    page.col_dim,
                    page.page_id,
                    page,
                    shape_stats(page.shape_id)});
            });
            apply_plan = std::move(plan);
        }
        return *apply_plan;
    }

    void invalidate_apply_plan() const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        apply_plan.reset();
    }

    void record_apply_batch(int shape_id, ApplyMode mode, size_t block_count) const {
        auto* stats = shape_stats(shape_id);
        if (stats == nullptr) {
            return;
        }
        auto& entry = *stats;
        if (mode == ApplyMode::Batched) {
            entry.batched_batches.fetch_add(1, std::memory_order_relaxed);
            entry.batched_blocks.fetch_add(
                static_cast<uint64_t>(block_count),
                std::memory_order_relaxed);
            return;
        }
        entry.scalar_batches.fetch_add(1, std::memory_order_relaxed);
        entry.scalar_blocks.fetch_add(
            static_cast<uint64_t>(block_count),
            std::memory_order_relaxed);
    }

    void record_spmm_batch(
        int row_dim,
        int inner_dim,
        int col_dim,
        size_t product_count) const {
        auto* entry = ensure_spmm_policy(row_dim, inner_dim, col_dim);
        if (entry == nullptr) {
            return;
        }
        entry->launched_batches.fetch_add(1, std::memory_order_relaxed);
        entry->launched_products.fetch_add(
            static_cast<uint64_t>(product_count),
            std::memory_order_relaxed);
    }

    void reset_apply_counters() const {
        for (const auto& stats : shape_apply_stats) {
            if (!stats) {
                continue;
            }
            stats->scalar_batches.store(0, std::memory_order_relaxed);
            stats->scalar_blocks.store(0, std::memory_order_relaxed);
            stats->batched_batches.store(0, std::memory_order_relaxed);
            stats->batched_blocks.store(0, std::memory_order_relaxed);
        }
    }

    size_t shape_scalar_apply_batch_count(int shape_id) const {
        auto* stats = shape_stats(shape_id);
        return stats == nullptr
            ? 0
            : static_cast<size_t>(
                stats->scalar_batches.load(std::memory_order_relaxed));
    }

    size_t shape_batched_apply_batch_count(int shape_id) const {
        auto* stats = shape_stats(shape_id);
        return stats == nullptr
            ? 0
            : static_cast<size_t>(
                stats->batched_batches.load(std::memory_order_relaxed));
    }

    size_t total_scalar_apply_batch_count() const {
        size_t total = 0;
        for (const auto& stats : shape_apply_stats) {
            if (stats) {
                total += static_cast<size_t>(
                    stats->scalar_batches.load(std::memory_order_relaxed));
            }
        }
        return total;
    }

    size_t total_batched_apply_batch_count() const {
        size_t total = 0;
        for (const auto& stats : shape_apply_stats) {
            if (stats) {
                total += static_cast<size_t>(
                    stats->batched_batches.load(std::memory_order_relaxed));
            }
        }
        return total;
    }

    size_t spmm_batch_count(int row_dim, int inner_dim, int col_dim) const {
        const auto* policy = find_spmm_policy(row_dim, inner_dim, col_dim);
        return policy == nullptr
            ? 0
            : static_cast<size_t>(
                policy->launched_batches.load(std::memory_order_relaxed));
    }

    bool is_contiguous() const {
        return contiguous_layout;
    }

    void mark_noncontiguous() {
        contiguous_layout = false;
    }

    void pack_contiguous() {
        storage.rebuild_handles(blk_handles);
        contiguous_layout = true;
        invalidate_apply_plan();
    }

    template <typename Fn>
    void for_each_shape_class(Fn&& fn) const {
        storage.for_each_shape([&](const auto& record) {
            fn(
                record.shape_id,
                record.row_dim,
                record.col_dim,
                record.matrix_block_indices);
        });
    }

    template <typename Fn>
    void for_each_shape_batch(Fn&& fn) const {
        if constexpr (std::is_invocable_v<Fn, const ShapeBatchView&>) {
            for_each_shape_batch_view(std::forward<Fn>(fn));
        } else {
            for_each_shape_batch_metadata(std::forward<Fn>(fn));
        }
    }

    template <typename Fn>
    void for_each_shape_batch_view(Fn&& fn) const {
        const auto& plan = ensure_apply_plan();
        for (const auto& view : plan.batches) {
            fn(view);
        }
    }

    template <typename Fn>
    void for_each_shape_batch_metadata(Fn&& fn) const {
        const auto& plan = ensure_apply_plan();
        for (const auto& view : plan.batches) {
            // Compatibility adapter for older call sites that expect decomposed
            // batch metadata instead of the richer ShapeBatchView.
            std::vector<int> logical_slots;
            logical_slots.reserve(view.block_count());
            for (uint32_t idx = 0; idx < view.block_count(); ++idx) {
                logical_slots.push_back(view.logical_slot(idx));
            }
            fn(
                view.shape_id,
                view.row_dim,
                view.col_dim,
                view.page_id,
                logical_slots);
        }
    }

    T* get_ptr(uint64_t handle) {
        return storage.block_ptr(handle);
    }

    const T* get_ptr(uint64_t handle) const {
        return storage.block_ptr(handle);
    }

    size_t block_size_elements(int slot) const {
        return storage.elements_per_block(
            Storage::shape_id_of(blk_handles.at(static_cast<size_t>(slot))));
    }

private:
    ShapeApplyStats* ensure_shape_stats(int shape_id) {
        if (shape_id < 0) {
            throw std::logic_error("Negative VBCSR shape id");
        }
        if (static_cast<size_t>(shape_id) >= shape_apply_stats.size()) {
            shape_apply_stats.resize(static_cast<size_t>(shape_id) + 1);
        }
        if (!shape_apply_stats[static_cast<size_t>(shape_id)]) {
            shape_apply_stats[static_cast<size_t>(shape_id)] =
                std::make_unique<ShapeApplyStats>();
        }
        return shape_apply_stats[static_cast<size_t>(shape_id)].get();
    }

    ShapeApplyStats* shape_stats(int shape_id) const {
        if (shape_id < 0 || static_cast<size_t>(shape_id) >= shape_apply_stats.size()) {
            return nullptr;
        }
        return shape_apply_stats[static_cast<size_t>(shape_id)].get();
    }

    const SpMMStats* find_spmm_policy(
        int row_dim,
        int inner_dim,
        int col_dim) const {
        const SpMMPolicyKey key{row_dim, inner_dim, col_dim};
        std::lock_guard<std::mutex> lock(policy_mutex);
        auto it = spmm_policy_lookup.find(key);
        if (it == spmm_policy_lookup.end()) {
            return nullptr;
        }
        return spmm_stats_records[it->second].get();
    }

    SpMMStats* ensure_spmm_policy(
        int row_dim,
        int inner_dim,
        int col_dim) const {
        const SpMMPolicyKey key{row_dim, inner_dim, col_dim};
        std::lock_guard<std::mutex> lock(policy_mutex);
        auto it = spmm_policy_lookup.find(key);
        if (it != spmm_policy_lookup.end()) {
            return spmm_stats_records[it->second].get();
        }

        auto record = std::make_unique<SpMMStats>();
        record->row_dim = row_dim;
        record->inner_dim = inner_dim;
        record->col_dim = col_dim;
        const size_t record_index = spmm_stats_records.size();
        spmm_stats_records.push_back(std::move(record));
        spmm_policy_lookup.emplace(key, record_index);
        return spmm_stats_records.back().get();
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_VBCSR_BACKEND_HANDLE_HPP
