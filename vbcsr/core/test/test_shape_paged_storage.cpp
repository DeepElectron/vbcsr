#include "../detail/shape_paged_storage.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using vbcsr::detail::ShapeBlockStore;

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "test_shape_paged_storage: " << message << std::endl;
        return false;
    }
    return true;
}

} // namespace

int main() {
    using Storage = ShapeBlockStore<double>;

    Storage storage(2);
    if (!check(storage.max_blocks_per_page() == 2, "configured max blocks per page mismatch")) {
        return 1;
    }

    const int shape_a = storage.get_or_create_shape(2, 3, 3);
    const int shape_b = storage.get_or_create_shape(1, 1, 1);
    if (!check(shape_a == 0, "first shape id should be zero") ||
        !check(shape_b == 1, "second shape id should be one") ||
        !check(storage.shape_count() == 2, "shape count mismatch")) {
        return 1;
    }

    const uint64_t handle_a0 = storage.append(shape_a, 4);
    const uint64_t handle_a1 = storage.append(shape_a, 8);
    const uint64_t handle_a2 = storage.append(shape_a, 10);
    const uint64_t handle_b0 = storage.append(shape_b, 1);

    if (!check(Storage::shape_id_of(handle_a0) == shape_a, "shape decode failed") ||
        !check(Storage::page_id_of(handle_a0) == 0, "first page decode failed") ||
        !check(Storage::page_block_index_of(handle_a0) == 0, "first page block index decode failed") ||
        !check(Storage::page_id_of(handle_a1) == 0, "second page decode failed") ||
        !check(Storage::page_block_index_of(handle_a1) == 1, "second page block index decode failed") ||
        !check(Storage::page_id_of(handle_a2) == 1, "cross-page decode failed") ||
        !check(Storage::page_block_index_of(handle_a2) == 0, "cross-page page block index decode failed")) {
        return 1;
    }

    double* block_a0 = storage.block_ptr(handle_a0);
    double* block_a1 = storage.block_ptr(handle_a1);
    double* block_a2 = storage.block_ptr(handle_a2);
    double* block_b0 = storage.block_ptr(handle_b0);
    for (int i = 0; i < 6; ++i) {
        block_a0[i] = 10.0 + i;
        block_a1[i] = 20.0 + i;
        block_a2[i] = 30.0 + i;
    }
    block_b0[0] = 99.0;

    if (!check(storage.elements_per_block(shape_a) == 6, "shape A block size mismatch") ||
        !check(storage.row_dim(shape_a) == 2, "shape A row dim mismatch") ||
        !check(storage.col_dim(shape_a) == 3, "shape A col dim mismatch") ||
        !check(storage.block_count(shape_a) == 3, "shape A block count mismatch") ||
        !check(storage.block_count(shape_b) == 1, "shape B block count mismatch") ||
        !check(storage.scalar_value_count() == 19, "scalar value count mismatch")) {
        return 1;
    }

    int shape_entries = 0;
    storage.for_each_shape([&](const auto& entry) {
        ++shape_entries;
        if (entry.shape_id == shape_a) {
            check(entry.matrix_block_indices.size() == 3, "shape A matrix block index list mismatch");
            check(entry.matrix_block_indices[0] == 4, "shape A matrix block index order mismatch (0)");
            check(entry.matrix_block_indices[1] == 8, "shape A matrix block index order mismatch (1)");
            check(entry.matrix_block_indices[2] == 10, "shape A matrix block index order mismatch (2)");
        }
    });
    if (!check(shape_entries == 2, "for_each_shape count mismatch")) {
        return 1;
    }

    int shape_a_page_count = 0;
    storage.for_each_page([&](const Storage::ShapePage& page) {
        if (page.shape_id != shape_a) {
            return;
        }
        ++shape_a_page_count;
        if (page.page_id == 0) {
            check(page.block_count == 2, "shape A first page block count mismatch");
            check(page.blocks_per_page == 2, "shape A first page block capacity mismatch");
            check(page.matrix_block(0) == 4, "shape A first page matrix block index 0 mismatch");
            check(page.matrix_block(1) == 8, "shape A first page matrix block index 1 mismatch");
            check(std::abs(page.block_ptr(0)[0] - 10.0) < 1e-12, "shape A first page value 0 mismatch");
            check(std::abs(page.block_ptr(1)[0] - 20.0) < 1e-12, "shape A first page value 1 mismatch");
        } else if (page.page_id == 1) {
            check(page.block_count == 1, "shape A second page block count mismatch");
            check(page.first_shape_block == 2, "shape A second page first block index mismatch");
            check(page.matrix_block(0) == 10, "shape A second page matrix block index mismatch");
            check(std::abs(page.block_ptr(0)[0] - 30.0) < 1e-12, "shape A second page value mismatch");
        }
    });
    if (!check(shape_a_page_count == 2, "shape A page count mismatch")) {
        return 1;
    }

    std::vector<uint64_t> rebuilt_handles(12, 1234);
    storage.rebuild_handles(rebuilt_handles);
    if (!check(rebuilt_handles[4] == handle_a0, "rebuilt handle A0 mismatch") ||
        !check(rebuilt_handles[8] == handle_a1, "rebuilt handle A1 mismatch") ||
        !check(rebuilt_handles[10] == handle_a2, "rebuilt handle A2 mismatch") ||
        !check(rebuilt_handles[1] == handle_b0, "rebuilt handle B0 mismatch") ||
        !check(rebuilt_handles[0] == 0, "rebuilt handle unused slot should be zero")) {
        return 1;
    }

    std::cout << "test_shape_paged_storage passed" << std::endl;
    return 0;
}
