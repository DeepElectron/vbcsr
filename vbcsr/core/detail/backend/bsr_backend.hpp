#ifndef VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP
#define VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP

#include "backend_common.hpp"
#include "thread_domain.hpp"

namespace vbcsr::detail {

template <typename T>
struct BSRBlockSlice {
    const int* cols = nullptr;
    T* values = nullptr;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t block_value_count = 0;
    uint32_t page_index = 0;
    uint64_t first_block = 0;
};

template <typename T>
struct BSRPageBatch {
    const int* cols = nullptr;
    T* values = nullptr;
    const int* row_block_offsets = nullptr;
    uint32_t block_count = 0;
    uint32_t block_size = 0;
    uint32_t block_value_count = 0;
    uint32_t page_index = 0;
    uint64_t first_block = 0;
    int row_begin = 0;
    int row_end = 0;

    int row_count() const {
        return row_end - row_begin;
    }

    uint32_t row_block_start(int row) const {
        return static_cast<uint32_t>(
            row_block_offsets[static_cast<size_t>(row - row_begin)]);
    }

    uint32_t row_block_end(int row) const {
        return static_cast<uint32_t>(
            row_block_offsets[static_cast<size_t>(row - row_begin + 1)]);
    }

    const T* block_ptr(uint32_t local_block_index) const {
        return values + static_cast<size_t>(local_block_index) * block_value_count;
    }

    T* block_ptr(uint32_t local_block_index) {
        return values + static_cast<size_t>(local_block_index) * block_value_count;
    }
};

template <typename T>
struct BSRApplyBatchEntry {
    // Owning storage for batch.row_block_offsets.
    std::vector<int> row_block_offsets_storage;
    BSRPageBatch<const T> batch;

    BSRApplyBatchEntry() = default;
    BSRApplyBatchEntry(const BSRApplyBatchEntry&) = delete;
    BSRApplyBatchEntry& operator=(const BSRApplyBatchEntry&) = delete;

    BSRApplyBatchEntry(BSRApplyBatchEntry&& other) noexcept
        : row_block_offsets_storage(std::move(other.row_block_offsets_storage)),
          batch(other.batch) {
        batch.row_block_offsets =
            row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
    }

    BSRApplyBatchEntry& operator=(BSRApplyBatchEntry&& other) noexcept {
        if (this != &other) {
            row_block_offsets_storage = std::move(other.row_block_offsets_storage);
            batch = other.batch;
            batch.row_block_offsets =
                row_block_offsets_storage.empty() ? nullptr : row_block_offsets_storage.data();
        }
        return *this;
    }
};

template <typename T>
struct BSRApplyPlan {
    std::vector<BSRApplyBatchEntry<T>> batches;
};

// The apply structure transposed: for every block COLUMN, which blocks land
// in it and which row each came from. Built once per structure from the
// forward plan.
//
// This is what lets an adjoint apply partition by OUTPUT column. Every column
// is written by exactly one iteration, so threads never collide whatever the
// schedule, no thread needs a private copy of y, and the result does not
// depend on how many threads ran.
//
// The row-driven form this replaced gave each thread a full-size private y and
// merged them afterwards. That costs threads x |y| of scratch and a merge pass
// over all of it, so its memory AND its added work both grow with the thread
// count -- an adjoint apply could take longer on many threads than on one.
// The equivalent VBCSR kernel already worked by column; this is that
// algorithm, kept on the fixed-size BSR block kernels.
template <typename T>
struct BSRAdjointApplyPlan {
    std::vector<int> incoming_ptr;          // block_count + 1, CSR over columns
    std::vector<int> incoming_rows;         // one entry per block-nonzero
    std::vector<const T*> incoming_blocks;  // one entry per block-nonzero
    int block_count = 0;
};

} // namespace vbcsr::detail

#include "bsr_vendor_cache.hpp"

namespace vbcsr::detail {

template <typename T>
struct BSRMatrixBackend {
    static uint32_t max_blocks_per_page(int uniform_block_size) {
        if (uniform_block_size <= 0) {
            return std::numeric_limits<uint32_t>::max();
        }
        const uint64_t block_values =
            static_cast<uint64_t>(uniform_block_size) * static_cast<uint64_t>(uniform_block_size);
        const uint64_t index_limit =
            static_cast<uint64_t>(std::numeric_limits<int>::max());
        const uint64_t by_values =
            block_values == 0
                ? 1u
                : static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) /
                    block_values;
        const uint64_t bounded =
            std::max<uint64_t>(1, std::min<uint64_t>(by_values, index_limit));
        return static_cast<uint32_t>(bounded);
    }

    static uint32_t normalize_blocks_per_page(
        uint32_t requested,
        int uniform_block_size) {
        if (requested == 0) {
            return max_blocks_per_page(uniform_block_size);
        }
        if (uniform_block_size <= 0) {
            return std::max<uint32_t>(requested, 1u);
        }
        return static_cast<uint32_t>(std::clamp<uint64_t>(
            requested,
            1u,
            static_cast<uint64_t>(max_blocks_per_page(uniform_block_size))));
    }

    int block_size = 0;
    PagedBuffer<T> values;
    // Stored thread work split (see thread_domain.hpp): computed once from the
    // structure — lazily under apply_plan_mutex, or eagerly at construction —
    // and kept stable so storage first-touch placement stays consistent with
    // apply access. Cleared with the apply plan on structure changes.
    mutable ThreadDomainPartition thread_domains;
    mutable std::unique_ptr<BSRApplyPlan<T>> apply_plan;
    mutable std::unique_ptr<BSRAdjointApplyPlan<T>> adjoint_apply_plan;
    mutable std::mutex apply_plan_mutex;
    mutable std::unique_ptr<BSRVendorCache<T>> vendor_cache;
    mutable std::mutex vendor_cache_mutex;
    mutable std::atomic<uint64_t> vendor_launch_count{0};
    uint32_t configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();

    BSRMatrixBackend() = default;

    BSRMatrixBackend(int uniform_block_size, uint32_t blocks_per_page)
        : block_size(uniform_block_size),
          configured_blocks_per_page_(
              normalize_blocks_per_page(blocks_per_page, uniform_block_size)),
          values(std::max<uint32_t>(
              configured_blocks_per_page_ *
                  static_cast<uint32_t>(
                      uniform_block_size * uniform_block_size),
              1u)) {}

    BSRMatrixBackend(const BSRMatrixBackend&) = delete;
    BSRMatrixBackend& operator=(const BSRMatrixBackend&) = delete;

    BSRMatrixBackend(BSRMatrixBackend&& other) noexcept
        : block_size(other.block_size),
          values(std::move(other.values)),
          thread_domains(std::move(other.thread_domains)),
          configured_blocks_per_page_(other.configured_blocks_per_page_) {
        other.block_size = 0;
        other.configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();
        other.vendor_launch_count.store(0, std::memory_order_release);
    }

    BSRMatrixBackend& operator=(BSRMatrixBackend&& other) noexcept {
        if (this != &other) {
            block_size = other.block_size;
            values = std::move(other.values);
            configured_blocks_per_page_ = other.configured_blocks_per_page_;
            invalidate_apply_plan();
            // After the invalidate: the moved-in partition stays valid for the
            // moved-in structure.
            thread_domains = std::move(other.thread_domains);
            vendor_launch_count.store(0, std::memory_order_release);
            other.block_size = 0;
            other.configured_blocks_per_page_ = std::numeric_limits<uint32_t>::max();
            other.vendor_launch_count.store(0, std::memory_order_release);
        }
        return *this;
    }

    uint32_t configured_blocks_per_page() const {
        return configured_blocks_per_page_;
    }

    uint32_t active_blocks_per_page() const {
        return static_cast<uint32_t>(
            values.elements_per_page() / std::max<size_t>(values_per_block(), 1));
    }

    uint64_t scalar_value_count() const {
        return values.size();
    }

    size_t values_per_block() const {
        return static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
    }

    size_t block_count() const {
        const size_t values_in_block = values_per_block();
        return values_in_block == 0
            ? 0
            : static_cast<size_t>(values.size()) / values_in_block;
    }

    // Hands back the storage under every block below `block_index`. For an
    // in-place product consuming its own left operand row by row: those blocks
    // will never be read again, and on the mmap path this is a munmap, so the
    // memory leaves the process rather than sitting in a free list. Blocks at
    // or above the boundary keep their addresses. Returns bytes released.
    uint64_t release_blocks_before(uint64_t block_index) {
        const size_t per_block = values_per_block();
        if (per_block == 0) return 0;
        return values.release_pages_before(block_index * static_cast<uint64_t>(per_block));
    }

    void initialize_structure(uint64_t logical_blocks, int uniform_block_size) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(configured_blocks_per_page_, block_size);
        const uint32_t blocks_per_page = logical_blocks == 0
            ? configured_blocks_per_page_
            : static_cast<uint32_t>(
                  std::min<uint64_t>(logical_blocks, configured_blocks_per_page_));
        values = PagedBuffer<T>(std::max<uint32_t>(
            blocks_per_page * static_cast<uint32_t>(values_per_block()),
            1u));
        invalidate_apply_plan();
        values.resize(
            logical_blocks * static_cast<uint64_t>(values_per_block()));
    }

    void initialize_structure(
        uint64_t logical_blocks,
        int uniform_block_size,
        uint32_t blocks_per_page) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(blocks_per_page, uniform_block_size);
        invalidate_apply_plan();
        initialize_structure(logical_blocks, uniform_block_size);
    }

    // Same sizing as initialize_structure(row_ptr.back(), ...), but value
    // pages are zero-filled inside a parallel region, each domain by the
    // thread that owns it in the stored partition ("first touch"): on a NUMA
    // host the OS places each 4 KiB page on the node of the writing thread,
    // so the blocks a thread later reads in the apply are node-local. For
    // scalar types whose array-new is not trivial (std::complex), the
    // allocation itself touches first and placement follows the allocating
    // thread.
    void initialize_structure_first_touch(
        const std::vector<int>& row_ptr,
        int uniform_block_size,
        uint32_t blocks_per_page) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(blocks_per_page, uniform_block_size);
        const uint64_t logical_blocks =
            row_ptr.empty() ? 0 : static_cast<uint64_t>(row_ptr.back());
        const uint32_t page_blocks = logical_blocks == 0
            ? configured_blocks_per_page_
            : static_cast<uint32_t>(
                  std::min<uint64_t>(logical_blocks, configured_blocks_per_page_));
        values = PagedBuffer<T>(std::max<uint32_t>(
            page_blocks * static_cast<uint32_t>(values_per_block()),
            1u));
        invalidate_apply_plan();
        values.resize_uninitialized(
            logical_blocks * static_cast<uint64_t>(values_per_block()));

        const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
        thread_domains = build_thread_domain_partition(
            n_rows,
            thread_domain_max_threads(),
            [&](int row) { return row_ptr[row + 1] - row_ptr[row]; });
        if (row_ptr.empty()) {
            return;
        }
        const uint64_t block_values = static_cast<uint64_t>(values_per_block());
        // Static schedule maps domain d to thread d when the team is full; a
        // smaller team still zeroes every element (multiple domains per
        // thread), only losing locality.
        #pragma omp parallel for schedule(static)
        for (int domain = 0; domain < thread_domains.thread_count; ++domain) {
            values.zero_fill_range(
                static_cast<uint64_t>(row_ptr[thread_domains.domain_begin(domain)]) * block_values,
                static_cast<uint64_t>(row_ptr[thread_domains.domain_end(domain)]) * block_values);
        }
    }

    // Structure and thread domains exactly as initialize_structure_first_touch
    // builds them, but WITHOUT its zero pass over the whole buffer.
    //
    // That pass exists for first-touch NUMA placement, not for correctness of
    // the allocation itself, and it has a cost the paged store was designed to
    // avoid: it faults in every page of the result before a single value is
    // computed. On an SpGEMM result that is the entire answer resident up front,
    // which makes any attempt to release the INPUT as it is consumed pointless
    // -- the peak has already happened.
    //
    // The caller takes on the obligation the pass used to discharge: zero each
    // row range before accumulating into it, through zero_row_range below, and
    // do it from the same domain thread so the placement is unchanged.
    void initialize_structure_deferred_touch(
        const std::vector<int>& row_ptr,
        int uniform_block_size,
        uint32_t blocks_per_page) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(blocks_per_page, uniform_block_size);
        const uint64_t logical_blocks =
            row_ptr.empty() ? 0 : static_cast<uint64_t>(row_ptr.back());
        const uint32_t page_blocks = logical_blocks == 0
            ? configured_blocks_per_page_
            : static_cast<uint32_t>(
                  std::min<uint64_t>(logical_blocks, configured_blocks_per_page_));
        values = PagedBuffer<T>(std::max<uint32_t>(
            page_blocks * static_cast<uint32_t>(values_per_block()),
            1u));
        invalidate_apply_plan();
        values.resize_uninitialized(
            logical_blocks * static_cast<uint64_t>(values_per_block()));

        const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
        thread_domains = build_thread_domain_partition(
            n_rows,
            thread_domain_max_threads(),
            [&](int row) { return row_ptr[row + 1] - row_ptr[row]; });
    }

    // Zeroes the values of rows [row_begin, row_end). Call from the thread that
    // will write them, so first touch still lands where the apply partition
    // expects it.
    void zero_row_range(const std::vector<int>& row_ptr, int row_begin, int row_end) {
        if (row_begin >= row_end || row_ptr.empty()) return;
        const uint64_t block_values = static_cast<uint64_t>(values_per_block());
        values.zero_fill_range(
            static_cast<uint64_t>(row_ptr[row_begin]) * block_values,
            static_cast<uint64_t>(row_ptr[row_end]) * block_values);
    }

    void initialize_structure_for_complete_overwrite(
        uint64_t logical_blocks,
        int uniform_block_size,
        uint32_t configured_blocks_per_page,
        uint32_t active_blocks_per_page) {
        block_size = uniform_block_size;
        configured_blocks_per_page_ =
            normalize_blocks_per_page(configured_blocks_per_page, block_size);
        const uint32_t normalized_active_blocks = logical_blocks == 0
            ? configured_blocks_per_page_
            : static_cast<uint32_t>(
                  std::clamp<uint64_t>(
                      static_cast<uint64_t>(std::max<uint32_t>(active_blocks_per_page, 1u)),
                      1u,
                      std::min<uint64_t>(
                          logical_blocks,
                          static_cast<uint64_t>(configured_blocks_per_page_))));
        const uint64_t page_values =
            static_cast<uint64_t>(normalized_active_blocks) *
            static_cast<uint64_t>(values_per_block());
        if (page_values > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
            throw std::overflow_error("BSR complete-overwrite page is too large");
        }
        values = PagedBuffer<T>(std::max<uint32_t>(
            static_cast<uint32_t>(page_values),
            1u));
        invalidate_apply_plan();
        values.resize_uninitialized(
            logical_blocks * static_cast<uint64_t>(values_per_block()));
    }

    T* block_ptr(int slot) {
        return values.element_ptr(
            static_cast<uint64_t>(slot) *
            static_cast<uint64_t>(values_per_block()));
    }

    const T* block_ptr(int slot) const {
        return values.element_ptr(
            static_cast<uint64_t>(slot) *
            static_cast<uint64_t>(values_per_block()));
    }

    BSRBlockSlice<T> page(IndexSpan col_ind, uint32_t page_index) {
        auto value_page = values.page(page_index);
        const uint32_t block_value_count =
            static_cast<uint32_t>(this->values_per_block());
        const uint64_t first_block =
            value_page.first_element / static_cast<uint64_t>(block_value_count);
        const uint32_t block_count = value_page.count / block_value_count;
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRBlockSlice<T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            block_value_count,
            page_index,
            first_block};
    }

    BSRBlockSlice<const T> page(
        IndexSpan col_ind,
        uint32_t page_index) const {
        auto value_page = values.page(page_index);
        const uint32_t block_value_count =
            static_cast<uint32_t>(this->values_per_block());
        const uint64_t first_block =
            value_page.first_element / static_cast<uint64_t>(block_value_count);
        const uint32_t block_count = value_page.count / block_value_count;
        if (first_block + block_count > static_cast<uint64_t>(col_ind.size())) {
            throw std::out_of_range("BSRMatrixBackend::page column span out of bounds");
        }
        return BSRBlockSlice<const T>{
            col_ind.data() + first_block,
            value_page.data,
            block_count,
            static_cast<uint32_t>(block_size),
            block_value_count,
            page_index,
            first_block};
    }

    // Stored thread work split, weighted by row nnz. Computed once; kept
    // stable across applies so (stage B) storage placement first-touched
    // against it stays consistent. A parallel region with a different thread
    // count falls back to the dynamic split (see thread_domain_range in
    // thread_domain.hpp).
    const ThreadDomainPartition& ensure_thread_domains(
        const std::vector<int>& row_ptr) const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (thread_domains.empty()) {
            const int n_rows =
                row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
            thread_domains = build_thread_domain_partition(
                n_rows,
                thread_domain_max_threads(),
                [&](int row) { return row_ptr[row + 1] - row_ptr[row]; });
        }
        return thread_domains;
    }

    const BSRApplyPlan<T>& ensure_apply_plan(
        const std::vector<int>& row_ptr,
        IndexSpan col_ind) const {
        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (apply_plan == nullptr) {
            auto plan = std::make_unique<BSRApplyPlan<T>>();
            plan->batches.reserve(values.page_count());

            for (uint32_t page_index = 0; page_index < values.page_count(); ++page_index) {
                const auto page_slice = page(col_ind, page_index);
                if (page_slice.block_count == 0) {
                    continue;
                }

                BSRApplyBatchEntry<T> batch_entry;
                batch_entry.batch.cols = page_slice.cols;
                batch_entry.batch.values = page_slice.values;
                batch_entry.batch.block_count = page_slice.block_count;
                batch_entry.batch.block_size = page_slice.block_size;
                batch_entry.batch.block_value_count = page_slice.block_value_count;
                batch_entry.batch.page_index = page_slice.page_index;
                batch_entry.batch.first_block = page_slice.first_block;

                const int begin = static_cast<int>(page_slice.first_block);
                const int end = begin + static_cast<int>(page_slice.block_count);
                const PageRowSpan row_span = find_page_row_span(row_ptr, begin, end);
                batch_entry.batch.row_begin = row_span.row_begin;
                batch_entry.batch.row_end = row_span.row_end;
                batch_entry.row_block_offsets_storage.reserve(
                    static_cast<size_t>(row_span.row_count() + 1));
                emit_page_local_row_ptr(
                    row_ptr,
                    begin,
                    end,
                    row_span,
                    [&](int page_local_offset) {
                        batch_entry.row_block_offsets_storage.push_back(page_local_offset);
                    });
                batch_entry.batch.row_block_offsets =
                    batch_entry.row_block_offsets_storage.data();
                plan->batches.push_back(std::move(batch_entry));
            }

            apply_plan = std::move(plan);
        }
        return *apply_plan;
    }

    /// The column-oriented transpose of the apply plan; see
    /// BSRAdjointApplyPlan. `block_count` counts owned AND ghost blocks,
    /// because an adjoint writes ghost columns before reducing them.
    const BSRAdjointApplyPlan<T>& ensure_adjoint_apply_plan(
        const std::vector<int>& row_ptr,
        IndexSpan col_ind,
        int block_count) const {
        // Outside the lock: ensure_apply_plan takes the same mutex, which is
        // not recursive.
        const BSRApplyPlan<T>& forward = ensure_apply_plan(row_ptr, col_ind);

        std::lock_guard<std::mutex> lock(apply_plan_mutex);
        if (adjoint_apply_plan != nullptr &&
            adjoint_apply_plan->block_count == block_count) {
            return *adjoint_apply_plan;
        }

        auto plan = std::make_unique<BSRAdjointApplyPlan<T>>();
        plan->block_count = block_count;
        plan->incoming_ptr.assign(static_cast<size_t>(block_count) + 1, 0);

        // Counting pass, then prefix sum, then a placement pass -- the usual
        // CSR transpose. Both passes walk the batches by row so they visit
        // exactly the same blocks.
        for (const auto& batch_entry : forward.batches) {
            const auto& batch = batch_entry.batch;
            for (int row = batch.row_begin; row < batch.row_end; ++row) {
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    ++plan->incoming_ptr[static_cast<size_t>(batch.cols[local_block]) + 1];
                }
            }
        }
        for (int col = 0; col < block_count; ++col) {
            plan->incoming_ptr[static_cast<size_t>(col) + 1] +=
                plan->incoming_ptr[static_cast<size_t>(col)];
        }

        const size_t block_nnz =
            static_cast<size_t>(plan->incoming_ptr[static_cast<size_t>(block_count)]);
        plan->incoming_rows.assign(block_nnz, 0);
        plan->incoming_blocks.assign(block_nnz, nullptr);

        std::vector<int> cursor(plan->incoming_ptr.begin(), plan->incoming_ptr.end() - 1);
        for (const auto& batch_entry : forward.batches) {
            const auto& batch = batch_entry.batch;
            for (int row = batch.row_begin; row < batch.row_end; ++row) {
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    const int dest = cursor[static_cast<size_t>(col)]++;
                    plan->incoming_rows[static_cast<size_t>(dest)] = row;
                    // A pointer into the value pages, exactly as the forward
                    // batches hold: any change that moves them also calls
                    // invalidate_apply_plan(), which drops this plan.
                    plan->incoming_blocks[static_cast<size_t>(dest)] =
                        batch.block_ptr(local_block);
                }
            }
        }

        adjoint_apply_plan = std::move(plan);
        return *adjoint_apply_plan;
    }

    const BSRVendorCache<T>& ensure_vendor_cache(
        const std::vector<int>& row_ptr,
        IndexSpan col_ind,
        int num_block_cols) const {
        {
            std::lock_guard<std::mutex> lock(vendor_cache_mutex);
            if (vendor_cache != nullptr) {
                return *vendor_cache;
            }
        }

        const auto& plan = ensure_apply_plan(row_ptr, col_ind);
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache == nullptr) {
            auto cache = std::make_unique<BSRVendorCache<T>>();
            // The backend keeps the persistent cache tied to storage lifetime;
            // the helper builds vendor descriptors from the already prepared
            // apply-plan batches.
            build_bsr_vendor_cache(*cache, plan, num_block_cols);
            vendor_cache = std::move(cache);
        }
        return *vendor_cache;
    }

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    bool ensure_mkl_mm_handles(
        const BSRVendorCache<T>& cache,
        int num_rhs) const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (cache.kind != BSRVendorBackendKind::MKL ||
            vendor_cache == nullptr ||
            vendor_cache.get() != &cache) {
            return false;
        }

        // Backend locking protects cache mutation; vendor helpers do the
        // MKL-specific MM descriptor construction.
        return ensure_bsr_mkl_mm_handles(*vendor_cache, num_rhs);
    }
#endif

private:
    void invalidate_apply_plan() const {
        {
            std::lock_guard<std::mutex> lock(apply_plan_mutex);
            apply_plan.reset();
            adjoint_apply_plan.reset();
            thread_domains = ThreadDomainPartition{};
        }
        invalidate_vendor_cache();
    }

    void invalidate_vendor_cache() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        vendor_cache.reset();
        vendor_launch_count.store(0, std::memory_order_release);
    }

public:
    // Inspection and instrumentation helpers used by tests, benchmarks, and
    // vendor dispatch counters. They are kept at the end of the class so the
    // storage/build logic stays easy to read top-to-bottom.
    BSRVendorBackendKind vendor_backend_kind() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        if (vendor_cache != nullptr) {
            return vendor_cache->kind;
        }
        return preferred_bsr_vendor_backend<T>();
    }

    std::string vendor_backend_name() const {
        return bsr_vendor_backend_name(vendor_backend_kind());
    }

    const void* vendor_cache_identity() const {
        std::lock_guard<std::mutex> lock(vendor_cache_mutex);
        return vendor_cache.get();
    }

    uint64_t get_vendor_launch_count() const {
        return vendor_launch_count.load(std::memory_order_acquire);
    }

    void reset_vendor_launch_count() const {
        vendor_launch_count.store(0, std::memory_order_release);
    }

    void note_vendor_launch(uint64_t batch_calls = 1) const {
        vendor_launch_count.fetch_add(batch_calls, std::memory_order_acq_rel);
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_BSR_BACKEND_HPP
