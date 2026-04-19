#ifndef VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP

#include "backend_common.hpp"

namespace vbcsr::detail {

enum class VBCSRApplyMode {
    Scalar,
    Batched
};

struct VBCSRApplyStats {
    std::atomic<uint64_t> scalar_batches{0};
    std::atomic<uint64_t> batched_batches{0};
};

template <typename T>
struct VBCSRPageBatch {
    int shape_id = -1;
    int row_dim = 0;
    int col_dim = 0;
    int page_id = -1;
    const T* values = nullptr;
    const int* graph_block_indices = nullptr;
    const int* graph_block_rows = nullptr;
    const int* graph_block_cols = nullptr;
    uint32_t live_block_count = 0;
    uint32_t blocks_per_page = 0;
    size_t block_value_count = 0;
    const VBCSRApplyStats* apply_stats = nullptr;

    uint32_t block_count() const {
        return live_block_count;
    }

    uint32_t block_capacity() const {
        return blocks_per_page;
    }

    size_t values_per_block() const {
        return block_value_count;
    }

    int graph_block_index(uint32_t block_index) const {
        if (block_index >= live_block_count) {
            throw std::out_of_range("VBCSRPageBatch graph block index out of bounds");
        }
        return graph_block_indices[block_index];
    }

    int row_block_index(uint32_t block_index) const {
        const int graph_block = graph_block_index(block_index);
        if (graph_block_rows == nullptr) {
            throw std::logic_error("VBCSRPageBatch row-block metadata is unavailable");
        }
        return graph_block_rows[graph_block];
    }

    int col_block_index(uint32_t block_index) const {
        const int graph_block = graph_block_index(block_index);
        if (graph_block_cols == nullptr) {
            throw std::logic_error("VBCSRPageBatch col-block metadata is unavailable");
        }
        return graph_block_cols[graph_block];
    }

    const T* block_ptr(uint32_t block_index) const {
        if (block_index >= live_block_count) {
            throw std::out_of_range("VBCSRPageBatch block pointer index out of bounds");
        }
        return values + static_cast<size_t>(block_index) * block_value_count;
    }

    uint64_t scalar_apply_batch_count() const {
        return apply_stats == nullptr
            ? 0
            : apply_stats->scalar_batches.load(std::memory_order_relaxed);
    }

    uint64_t batched_apply_batch_count() const {
        return apply_stats == nullptr
            ? 0
            : apply_stats->batched_batches.load(std::memory_order_relaxed);
    }
};

template <typename T>
struct VBCSRApplyPlan {
    std::vector<int> graph_block_rows_storage;
    std::vector<VBCSRPageBatch<T>> batches;
};

template <typename T, typename Kernel>
struct VBCSRMatrixBackend {
    using Storage = ShapeBlockStore<T>;
    using ApplyMode = VBCSRApplyMode;
    using ApplyStats = VBCSRApplyStats;
    using PageBatch = VBCSRPageBatch<T>;
    using ApplyPlan = VBCSRApplyPlan<T>;

    // "graph block index" means the flat local block position from row_ptr/col_ind.
    // Storage capacity and page traversal are block-oriented.
    static constexpr uint32_t hard_safe_blocks_per_page() {
        return Storage::hard_safe_blocks_per_page();
    }

    static uint32_t normalize_blocks_per_page(uint32_t requested) {
        if (requested == 0) {
            return hard_safe_blocks_per_page();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(
                requested,
                1u,
                static_cast<uint64_t>(hard_safe_blocks_per_page())));
    }

    VBCSRMatrixBackend() = default;

    explicit VBCSRMatrixBackend(uint32_t blocks_per_page)
        : storage(blocks_per_page) {}

    VBCSRMatrixBackend(const VBCSRMatrixBackend&) = delete;
    VBCSRMatrixBackend& operator=(const VBCSRMatrixBackend&) = delete;

    VBCSRMatrixBackend(VBCSRMatrixBackend&& other) noexcept
        : graph_block_handles_(std::move(other.graph_block_handles_)),
          storage(std::move(other.storage)),
          apply_stats_by_shape(std::move(other.apply_stats_by_shape)) {}

    VBCSRMatrixBackend& operator=(VBCSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            graph_block_handles_ = std::move(other.graph_block_handles_);
            storage = std::move(other.storage);
            apply_stats_by_shape = std::move(other.apply_stats_by_shape);
            invalidate_apply_plan();
        }
        return *this;
    }

    size_t local_scalar_nnz() const {
        return storage.scalar_value_count();
    }

    int shape_class_count() const {
        return storage.shape_count();
    }

    // This is the configured global cap forwarded to ShapeBlockStore. Each shape
    // may still realize a smaller page size, and existing pages keep their current
    // capacity until the backend is rebuilt.
    uint32_t configured_blocks_per_page() const {
        return storage.max_blocks_per_page();
    }

    int ensure_shape(int row_dim, int col_dim, size_t expected_block_count = 0) {
        const int shape_id =
            storage.get_or_create_shape(row_dim, col_dim, expected_block_count);
        ensure_apply_stats(shape_id);
        invalidate_apply_plan();
        return shape_id;
    }

    void initialize_graph_block_handles(size_t graph_block_count) {
        graph_block_handles_.resize(graph_block_count);
    }

    uint64_t graph_block_handle(int graph_block_index) const {
        return graph_block_handles_.at(static_cast<size_t>(graph_block_index));
    }

    void set_graph_block_handle(int graph_block_index, uint64_t handle) {
        graph_block_handles_.at(static_cast<size_t>(graph_block_index)) = handle;
    }

    int shape_id_for_graph_block(int graph_block_index) const {
        return Storage::shape_id_of(graph_block_handle(graph_block_index));
    }

    uint64_t append_block_for_shape(int shape_id, int graph_block_index) {
        const uint64_t handle = storage.append(shape_id, graph_block_index);
        invalidate_apply_plan();
        return handle;
    }

    template <typename Fn>
    void for_each_shape_class(Fn&& fn) const {
        storage.for_each_shape([&](const auto& record) {
            fn(
                record.shape_id,
                record.row_dim,
                record.col_dim,
                record.graph_block_indices);
        });
    }

    T* block_ptr_for_graph_block(int graph_block_index) {
        return block_ptr_from_handle(graph_block_handle(graph_block_index));
    }

    const T* block_ptr_for_graph_block(int graph_block_index) const {
        return block_ptr_from_handle(graph_block_handle(graph_block_index));
    }

    size_t block_size_elements_for_graph_block(int graph_block_index) const {
        return storage.elements_per_block(shape_id_for_graph_block(graph_block_index));
    }

    const ApplyPlan& ensure_apply_plan(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind) const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (!apply_plan) {
            auto plan = std::make_unique<ApplyPlan>();
            plan->graph_block_rows_storage.assign(col_ind.size(), 0);
            int row = 0;
            for (size_t graph_block_index = 0; graph_block_index < col_ind.size(); ++graph_block_index) {
                while (static_cast<size_t>(row + 1) < row_ptr.size() &&
                       graph_block_index >= static_cast<size_t>(row_ptr[static_cast<size_t>(row + 1)])) {
                    ++row;
                }
                plan->graph_block_rows_storage[graph_block_index] = row;
            }
            storage.for_each_page([&](const typename Storage::ShapePage& page) {
                plan->batches.push_back(PageBatch{
                    page.shape_id,
                    page.row_dim,
                    page.col_dim,
                    page.page_id,
                    page.data,
                    page.graph_block_indices,
                    plan->graph_block_rows_storage.data(),
                    col_ind.data(),
                    page.block_count,
                    page.blocks_per_page,
                    page.elements_per_block,
                    apply_stats_for_shape(page.shape_id)});
            });
            apply_plan = std::move(plan);
        }
        return *apply_plan;
    }

    template <typename Fn>
    void for_each_shape_batch(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        Fn&& fn) const {
        const auto& plan = ensure_apply_plan(row_ptr, col_ind);
        for (const auto& batch : plan.batches) {
            fn(batch);
        }
    }

    void record_apply_batch(int shape_id, ApplyMode mode) const {
        auto* stats = apply_stats_for_shape(shape_id);
        if (stats == nullptr) {
            return;
        }
        if (mode == ApplyMode::Batched) {
            stats->batched_batches.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        stats->scalar_batches.fetch_add(1, std::memory_order_relaxed);
    }

private:
    void invalidate_apply_plan() const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        apply_plan.reset();
    }

    T* block_ptr_from_handle(uint64_t handle) {
        return storage.block_ptr(handle);
    }

    const T* block_ptr_from_handle(uint64_t handle) const {
        return storage.block_ptr(handle);
    }

    ApplyStats* ensure_apply_stats(int shape_id) {
        if (shape_id < 0) {
            throw std::logic_error("Negative VBCSR shape id");
        }
        if (static_cast<size_t>(shape_id) >= apply_stats_by_shape.size()) {
            apply_stats_by_shape.resize(static_cast<size_t>(shape_id) + 1);
        }
        if (!apply_stats_by_shape[static_cast<size_t>(shape_id)]) {
            apply_stats_by_shape[static_cast<size_t>(shape_id)] =
                std::make_unique<ApplyStats>();
        }
        return apply_stats_by_shape[static_cast<size_t>(shape_id)].get();
    }

    ApplyStats* apply_stats_for_shape(int shape_id) const {
        if (shape_id < 0 || static_cast<size_t>(shape_id) >= apply_stats_by_shape.size()) {
            return nullptr;
        }
        return apply_stats_by_shape[static_cast<size_t>(shape_id)].get();
    }

    std::vector<uint64_t> graph_block_handles_;
    Storage storage;
    std::vector<std::unique_ptr<ApplyStats>> apply_stats_by_shape;
    mutable std::mutex apply_plan_mutex;
    mutable std::unique_ptr<ApplyPlan> apply_plan;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP
