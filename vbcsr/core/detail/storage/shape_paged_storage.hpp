#ifndef VBCSR_DETAIL_STORAGE_SHAPE_PAGED_STORAGE_HPP
#define VBCSR_DETAIL_STORAGE_SHAPE_PAGED_STORAGE_HPP

#include "paged_array.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vbcsr::detail {

template <typename T>
class ShapeBlockStore {
public:
    static constexpr size_t kDefaultPayloadPageElements = 1ULL << 22;
    static constexpr uint64_t kShapeBits = 16;
    static constexpr uint64_t kPageBits = 24;
    static constexpr uint64_t kPageBlockBits = 24;
    static constexpr uint64_t kShapeShift = kPageBits + kPageBlockBits;
    static constexpr uint64_t kPageShift = kPageBlockBits;
    static constexpr uint64_t kShapeMask = (uint64_t(1) << kShapeBits) - 1;
    static constexpr uint64_t kPageMask = (uint64_t(1) << kPageBits) - 1;
    static constexpr uint64_t kPageBlockMask = (uint64_t(1) << kPageBlockBits) - 1;

    struct ShapeRecord {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        size_t elements_per_block = 0;
        uint32_t blocks_per_page = 0;
        size_t used_blocks = 0;
        size_t reserved_blocks = 0;
        PagedBuffer<T> values; // use the PagedBuffer as internal storage
        std::vector<int> graph_block_indices;
    };

    struct ShapePage {
        int shape_id = -1;
        int row_dim = 0;
        int col_dim = 0;
        int page_id = -1;
        const T* data = nullptr;
        const int* graph_block_indices = nullptr;
        uint32_t block_count = 0;
        uint32_t blocks_per_page = 0;
        size_t elements_per_block = 0;
        uint64_t first_shape_block = 0;

        const T* block_ptr(uint32_t block_index) const {
            if (block_index >= block_count) {
                throw std::out_of_range("ShapeBlockStore::ShapePage block index out of bounds");
            }
            return data + static_cast<size_t>(block_index) * elements_per_block;
        }
    };

    ShapeBlockStore() = default;

    explicit ShapeBlockStore(uint32_t max_blocks_per_page)
        : max_blocks_per_page_(normalize_max_blocks_per_page(max_blocks_per_page)) {}

    uint32_t max_blocks_per_page() const {
        return max_blocks_per_page_;
    }

    void set_max_blocks_per_page(uint32_t max_blocks_per_page) {
        max_blocks_per_page_ = normalize_max_blocks_per_page(max_blocks_per_page);
    }

    static uint32_t hard_safe_blocks_per_page() {
        return static_cast<uint32_t>(kPageBlockMask);
    }

    int get_or_create_shape(int row_dim, int col_dim, size_t reserve_blocks = 0) {
        const auto key = std::make_pair(row_dim, col_dim);
        auto it = shape_lookup_.find(key);
        if (it != shape_lookup_.end()) {
            auto& record = require_shape(it->second);
            ensure_reserved_blocks(record, reserve_blocks);
            return it->second;
        }

        const int shape_id = static_cast<int>(shapes_.size());
        if (static_cast<uint64_t>(shape_id) > kShapeMask) {
            throw std::overflow_error("ShapeBlockStore shape registry exceeded handle capacity");
        }

        ShapeRecord record;
        record.shape_id = shape_id;
        record.row_dim = row_dim;
        record.col_dim = col_dim;
        record.elements_per_block = static_cast<size_t>(row_dim) * static_cast<size_t>(col_dim);
        record.blocks_per_page = effective_blocks_per_page(record.elements_per_block, reserve_blocks);
        record.values = PagedBuffer<T>(page_size_elements(record.elements_per_block, record.blocks_per_page));
        ensure_reserved_blocks(record, reserve_blocks); // same as above worry

        shapes_.push_back(std::move(record));
        shape_lookup_.emplace(key, shape_id);
        return shape_id;
    }

    uint64_t append(int shape_id, int graph_block_index) {
        if (graph_block_index < 0) {
            throw std::logic_error("ShapeBlockStore graph block index cannot be negative");
        }

        auto& record = require_shape(shape_id);
        if (record.used_blocks == record.reserved_blocks) {
            const size_t grown_blocks =
                record.reserved_blocks == 0
                    ? static_cast<size_t>(record.blocks_per_page)
                    : record.reserved_blocks * 2; // will this expode or create too many unused memory?
            ensure_reserved_blocks(record, std::max(record.used_blocks + 1, grown_blocks));
        }

        const size_t shape_block_index = record.used_blocks;
        ++record.used_blocks;
        record.graph_block_indices.push_back(graph_block_index);
        record.values.resize(live_value_count(record));
        return encode_handle(
            shape_id,
            page_id_from_shape_block(record, shape_block_index),
            page_block_index_from_shape_block(record, shape_block_index));
    }

    T* block_ptr(uint64_t handle) {
        return const_cast<T*>(static_cast<const ShapeBlockStore&>(*this).block_ptr(handle));
    }

    const T* block_ptr(uint64_t handle) const {
        const auto* record = find_shape(shape_id_of(handle));
        if (record == nullptr) {
            throw std::logic_error("ShapeBlockStore handle shape id out of bounds");
        }
        const size_t shape_block_index = shape_block_from_handle(*record, handle);
        if (shape_block_index >= record->used_blocks) {
            throw std::out_of_range("ShapeBlockStore handle points past live blocks");
        }
        return record->values.element_ptr(shape_block_index * record->elements_per_block);
    }

    static int shape_id_of(uint64_t handle) {
        return static_cast<int>((handle >> kShapeShift) & kShapeMask);
    }

    static int page_id_of(uint64_t handle) {
        return static_cast<int>((handle >> kPageShift) & kPageMask);
    }

    static uint32_t page_block_index_of(uint64_t handle) {
        return static_cast<uint32_t>(handle & kPageBlockMask);
    }

    size_t elements_per_block(int shape_id) const {
        return require_shape(shape_id).elements_per_block;
    }

    int row_dim(int shape_id) const {
        return require_shape(shape_id).row_dim;
    }

    int col_dim(int shape_id) const {
        return require_shape(shape_id).col_dim;
    }

    size_t block_count(int shape_id) const {
        return require_shape(shape_id).used_blocks;
    }

    int shape_count() const {
        return static_cast<int>(shapes_.size());
    }

    size_t scalar_value_count() const {
        size_t total = 0;
        for (const auto& record : shapes_) {
            total += live_value_count(record);
        }
        return total;
    }

    template <typename Fn>
    void for_each_shape(Fn&& fn) const {
        for (const auto& record : shapes_) {
            if (record.used_blocks == 0) {
                continue;
            }
            fn(record);
        }
    }

    template <typename Fn>
    void for_each_page(Fn&& fn) const {
        for (const auto& record : shapes_) {
            if (record.used_blocks == 0) {
                continue;
            }

            for (uint32_t page_index = 0; page_index < record.values.page_count(); ++page_index) {
                const uint64_t first_shape_block =
                    static_cast<uint64_t>(page_index) * static_cast<uint64_t>(record.blocks_per_page);
                if (first_shape_block >= record.used_blocks) {
                    break;
                }

                const auto value_page = record.values.page(page_index);
                const uint32_t visible_blocks = static_cast<uint32_t>(
                    std::min<uint64_t>(
                        static_cast<uint64_t>(value_page.count / std::max<size_t>(record.elements_per_block, 1)),
                        record.used_blocks - first_shape_block));
                if (visible_blocks == 0) {
                    continue;
                }

                fn(ShapePage{
                    record.shape_id,
                    record.row_dim,
                    record.col_dim,
                    static_cast<int>(page_index),
                    value_page.data,
                    record.graph_block_indices.data() + first_shape_block,
                    visible_blocks,
                    record.blocks_per_page,
                    record.elements_per_block,
                    first_shape_block});
            }
        }
    }

private:
    static uint32_t normalize_max_blocks_per_page(uint32_t requested) {
        if (requested == 0) {
            return hard_safe_blocks_per_page();
        }
        return static_cast<uint32_t>(
            std::clamp<uint64_t>(requested, 1u, static_cast<uint64_t>(hard_safe_blocks_per_page())));
    }

    static uint32_t default_blocks_per_page(size_t elements_per_block) {
        if (elements_per_block == 0) {
            return 1;
        }
        const size_t page_elements = std::max<size_t>(1, kDefaultPayloadPageElements / elements_per_block);
        const size_t bounded = std::min<size_t>(page_elements, static_cast<size_t>(kPageBlockMask));
        return static_cast<uint32_t>(std::max<size_t>(bounded, 1));
    }

    static uint32_t max_blocks_per_page_for_payload(size_t elements_per_block) {
        if (elements_per_block == 0) {
            return 1;
        }
        const uint64_t payload_limit =
            static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / static_cast<uint64_t>(elements_per_block);
        return static_cast<uint32_t>(
            std::max<uint64_t>(1, std::min<uint64_t>(payload_limit, static_cast<uint64_t>(kPageBlockMask))));
    }

    uint32_t effective_blocks_per_page(size_t elements_per_block, size_t reserve_blocks) const {
        const size_t requested_blocks =
            reserve_blocks == 0 ? default_blocks_per_page(elements_per_block) : reserve_blocks;
        const size_t bounded = std::min<size_t>(
            {requested_blocks,
             static_cast<size_t>(max_blocks_per_page_),
             static_cast<size_t>(hard_safe_blocks_per_page()),
             static_cast<size_t>(max_blocks_per_page_for_payload(elements_per_block))});
        return static_cast<uint32_t>(std::max<size_t>(bounded, 1));
    }

    static uint32_t page_size_elements(size_t elements_per_block, uint32_t blocks_per_page) {
        const uint64_t total =
            static_cast<uint64_t>(std::max<size_t>(elements_per_block, 1)) * static_cast<uint64_t>(blocks_per_page);
        if (total == 0 || total > std::numeric_limits<uint32_t>::max()) {
            throw std::overflow_error("ShapeBlockStore payload page size overflow");
        }
        return static_cast<uint32_t>(total);
    }

    static uint64_t live_value_count(const ShapeRecord& record) {
        return static_cast<uint64_t>(record.used_blocks) * static_cast<uint64_t>(record.elements_per_block);
    }

    void ensure_reserved_blocks(ShapeRecord& record, size_t reserve_blocks) {
        if (reserve_blocks <= record.reserved_blocks) {
            return;
        }
        record.values.reserve(
            static_cast<uint64_t>(reserve_blocks) * static_cast<uint64_t>(record.elements_per_block));
        record.reserved_blocks = reserve_blocks;
        record.graph_block_indices.reserve(reserve_blocks);
    }

    static int page_id_from_shape_block(const ShapeRecord& record, size_t shape_block_index) {
        return static_cast<int>(shape_block_index / record.blocks_per_page);
    }

    static uint32_t page_block_index_from_shape_block(const ShapeRecord& record, size_t shape_block_index) {
        return static_cast<uint32_t>(shape_block_index % record.blocks_per_page);
    }

    static size_t shape_block_from_handle(const ShapeRecord& record, uint64_t handle) {
        return static_cast<size_t>(page_id_of(handle)) * static_cast<size_t>(record.blocks_per_page) +
               static_cast<size_t>(page_block_index_of(handle));
    }

    static uint64_t encode_handle(int shape_id, int page_id, uint32_t page_block_index) {
        if (shape_id < 0 || page_id < 0) {
            throw std::logic_error("ShapeBlockStore handle cannot encode negative indices");
        }
        if (static_cast<uint64_t>(shape_id) > kShapeMask ||
            static_cast<uint64_t>(page_id) > kPageMask ||
            static_cast<uint64_t>(page_block_index) > kPageBlockMask) {
            throw std::overflow_error("ShapeBlockStore handle field overflow");
        }
        return (static_cast<uint64_t>(shape_id) << kShapeShift) |
               (static_cast<uint64_t>(page_id) << kPageShift) |
               static_cast<uint64_t>(page_block_index);
    }

    ShapeRecord& require_shape(int shape_id) {
        if (shape_id < 0 || shape_id >= static_cast<int>(shapes_.size())) {
            throw std::logic_error("Invalid ShapeBlockStore shape identifier");
        }
        return shapes_.at(static_cast<size_t>(shape_id));
    }

    const ShapeRecord& require_shape(int shape_id) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shapes_.size())) {
            throw std::logic_error("Invalid ShapeBlockStore shape identifier");
        }
        return shapes_.at(static_cast<size_t>(shape_id));
    }

    const ShapeRecord* find_shape(int shape_id) const {
        if (shape_id < 0 || shape_id >= static_cast<int>(shapes_.size())) {
            return nullptr;
        }
        return &shapes_[static_cast<size_t>(shape_id)];
    }

    uint32_t max_blocks_per_page_ = hard_safe_blocks_per_page();
    std::vector<ShapeRecord> shapes_;
    std::map<std::pair<int, int>, int> shape_lookup_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_STORAGE_SHAPE_PAGED_STORAGE_HPP
