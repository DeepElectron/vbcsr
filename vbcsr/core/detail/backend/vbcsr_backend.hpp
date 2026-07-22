#ifndef VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP

#include "backend_common.hpp"
#include "thread_domain.hpp"

namespace vbcsr::detail {

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
};

template <typename T>
struct VBCSRApplyPlan {
    std::vector<int> graph_block_rows_storage;
    std::vector<VBCSRPageBatch<T>> batches;
};

struct VBCSRForwardRowTask {
    int row = 0;
    int row_dim = 0;
    int block_begin = 0;
    int block_end = 0;
    int block_degree = 0;
    bool packed_output = false;
    size_t work = 0;
};

struct VBCSRForwardApplyPlan {
    int direct_dense_row_degree_limit = 0;
    int max_row_dim = 0;
    size_t total_work = 0;
    std::vector<VBCSRForwardRowTask> rows;
    // Stored work-balanced thread split over `rows`, computed once with the
    // plan (see thread_domain.hpp). Applies whose parallel region matches its
    // thread count use it directly; otherwise they rebuild dynamic chunks.
    ThreadDomainPartition thread_domains;
};

struct VBCSRAdjointColumnTask {
    int col = 0;
    int col_dim = 0;
    int incoming_begin = 0;
    int incoming_end = 0;
    int incoming_degree = 0;
    bool packed_output = false;
    size_t work = 0;
};

struct VBCSRAdjointApplyPlan {
    int direct_dense_col_degree_limit = 0;
    int block_count = 0;
    int max_col_dim = 0;
    size_t total_work = 0;
    std::vector<int> incoming_slots;
    std::vector<int> incoming_rows;
    std::vector<VBCSRAdjointColumnTask> columns;
    // Stored work-balanced thread split over `columns` (see
    // VBCSRForwardApplyPlan::thread_domains).
    ThreadDomainPartition thread_domains;
};

template <typename T>
struct VBCSRMatrixBackend {
    using Storage = ShapeBlockStore<T>;
    using PageBatch = VBCSRPageBatch<T>;
    using ApplyPlan = VBCSRApplyPlan<T>;
    using ForwardApplyPlan = VBCSRForwardApplyPlan;
    using ForwardRowTask = VBCSRForwardRowTask;
    using AdjointApplyPlan = VBCSRAdjointApplyPlan;
    using AdjointColumnTask = VBCSRAdjointColumnTask;

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
          storage(std::move(other.storage)) {}

    VBCSRMatrixBackend& operator=(VBCSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            graph_block_handles_ = std::move(other.graph_block_handles_);
            storage = std::move(other.storage);
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

    uint64_t append_block_for_shape_uninitialized(int shape_id, int graph_block_index) {
        const uint64_t handle = storage.append_uninitialized(shape_id, graph_block_index);
        invalidate_apply_plan();
        return handle;
    }

    void append_blocks_for_shape_uninitialized(
        int shape_id,
        const std::vector<int>& graph_block_indices) {
        storage.append_many_uninitialized(
            shape_id,
            graph_block_indices,
            [&](int graph_block_index, uint64_t handle) {
                set_graph_block_handle(graph_block_index, handle);
            });
        invalidate_apply_plan();
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

    // ShapeBlockStore keeps values packed by (shape, page), while apply kernels
    // still need graph-row and graph-column metadata for each packed block. The
    // apply plan is the cached bridge: graph_block_rows_storage reconstructs the
    // row for each flat graph block, and each PageBatch points at that row table,
    // the graph col_ind array, and the contiguous same-shape page payload.
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
                    page.elements_per_block});
            });
            apply_plan = std::move(plan);
        }
        return *apply_plan;
    }

    const ForwardApplyPlan& ensure_forward_apply_plan(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        const std::vector<int>& block_sizes,
        int n_rows,
        int direct_dense_row_degree_limit) const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (!forward_apply_plan ||
            forward_apply_plan->direct_dense_row_degree_limit != direct_dense_row_degree_limit ||
            static_cast<int>(forward_apply_plan->rows.size()) != n_rows) {
            auto plan = std::make_unique<ForwardApplyPlan>();
            plan->direct_dense_row_degree_limit = direct_dense_row_degree_limit;
            plan->rows.reserve(static_cast<size_t>(std::max(0, n_rows)));
            for (int row = 0; row < n_rows; ++row) {
                const int row_dim = block_sizes[row];
                const int block_begin = row_ptr[row];
                const int block_end = row_ptr[row + 1];
                const int block_degree = block_end - block_begin;
                size_t row_work = 0;
                for (int slot = block_begin; slot < block_end; ++slot) {
                    const int col = col_ind[slot];
                    row_work += static_cast<size_t>(row_dim) *
                                static_cast<size_t>(block_sizes[col]);
                }
                plan->max_row_dim = std::max(plan->max_row_dim, row_dim);
                plan->total_work += row_work;
                plan->rows.push_back(ForwardRowTask{
                    row,
                    row_dim,
                    block_begin,
                    block_end,
                    block_degree,
                    block_degree > direct_dense_row_degree_limit,
                    row_work});
            }
            plan->thread_domains = build_thread_domain_partition(
                n_rows,
                thread_domain_max_threads(),
                [&](int row) {
                    return std::max<size_t>(
                        size_t(1), plan->rows[static_cast<size_t>(row)].work);
                });
            forward_apply_plan = std::move(plan);
        }
        return *forward_apply_plan;
    }

    const AdjointApplyPlan& ensure_adjoint_apply_plan(
        const std::vector<int>& row_ptr,
        const std::vector<int>& col_ind,
        const std::vector<int>& block_sizes,
        int n_rows,
        int direct_dense_col_degree_limit) const {
        const int block_count = static_cast<int>(block_sizes.size());
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (!adjoint_apply_plan ||
            adjoint_apply_plan->direct_dense_col_degree_limit != direct_dense_col_degree_limit ||
            adjoint_apply_plan->block_count != block_count) {
            auto plan = std::make_unique<AdjointApplyPlan>();
            plan->direct_dense_col_degree_limit = direct_dense_col_degree_limit;
            plan->block_count = block_count;

            const int slot_count = n_rows > 0 ? row_ptr[static_cast<size_t>(n_rows)] : 0;
            std::vector<int> incoming_counts(static_cast<size_t>(std::max(0, block_count)), 0);
            for (int row = 0; row < n_rows; ++row) {
                for (int slot = row_ptr[row]; slot < row_ptr[row + 1]; ++slot) {
                    ++incoming_counts[static_cast<size_t>(col_ind[slot])];
                }
            }

            std::vector<int> incoming_offsets(static_cast<size_t>(block_count) + 1, 0);
            for (int col = 0; col < block_count; ++col) {
                incoming_offsets[static_cast<size_t>(col) + 1] =
                    incoming_offsets[static_cast<size_t>(col)] +
                    incoming_counts[static_cast<size_t>(col)];
            }

            plan->incoming_slots.assign(static_cast<size_t>(std::max(0, slot_count)), 0);
            plan->incoming_rows.assign(static_cast<size_t>(std::max(0, slot_count)), 0);
            auto cursor = incoming_offsets;
            for (int row = 0; row < n_rows; ++row) {
                for (int slot = row_ptr[row]; slot < row_ptr[row + 1]; ++slot) {
                    const int col = col_ind[slot];
                    const int dest = cursor[static_cast<size_t>(col)]++;
                    plan->incoming_slots[static_cast<size_t>(dest)] = slot;
                    plan->incoming_rows[static_cast<size_t>(dest)] = row;
                }
            }

            plan->columns.reserve(static_cast<size_t>(block_count));
            for (int col = 0; col < block_count; ++col) {
                const int incoming_begin = incoming_offsets[static_cast<size_t>(col)];
                const int incoming_end = incoming_offsets[static_cast<size_t>(col) + 1];
                if (incoming_begin == incoming_end) {
                    continue;
                }
                const int col_dim = block_sizes[col];
                size_t col_work = 0;
                for (int incoming = incoming_begin; incoming < incoming_end; ++incoming) {
                    const int row = plan->incoming_rows[static_cast<size_t>(incoming)];
                    col_work += static_cast<size_t>(block_sizes[row]) *
                                static_cast<size_t>(col_dim);
                }
                const int incoming_degree = incoming_end - incoming_begin;
                plan->max_col_dim = std::max(plan->max_col_dim, col_dim);
                plan->total_work += col_work;
                plan->columns.push_back(AdjointColumnTask{
                    col,
                    col_dim,
                    incoming_begin,
                    incoming_end,
                    incoming_degree,
                    incoming_degree > direct_dense_col_degree_limit,
                    col_work});
            }
            plan->thread_domains = build_thread_domain_partition(
                block_count,
                thread_domain_max_threads(),
                [&](int col) {
                    return std::max<size_t>(
                        size_t(1), plan->columns[static_cast<size_t>(col)].work);
                });
            adjoint_apply_plan = std::move(plan);
        }
        return *adjoint_apply_plan;
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

private:
    void invalidate_apply_plan() const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        apply_plan.reset();
        forward_apply_plan.reset();
        adjoint_apply_plan.reset();
    }

    T* block_ptr_from_handle(uint64_t handle) {
        return storage.block_ptr(handle);
    }

    const T* block_ptr_from_handle(uint64_t handle) const {
        return storage.block_ptr(handle);
    }

    std::vector<uint64_t> graph_block_handles_;
    Storage storage;
    mutable std::mutex apply_plan_mutex;
    mutable std::unique_ptr<ApplyPlan> apply_plan;
    mutable std::unique_ptr<ForwardApplyPlan> forward_apply_plan;
    mutable std::unique_ptr<AdjointApplyPlan> adjoint_apply_plan;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_VBCSR_BACKEND_HPP
