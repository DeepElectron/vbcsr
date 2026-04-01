#ifndef VBCSR_DETAIL_BACKEND_HANDLE_HPP
#define VBCSR_DETAIL_BACKEND_HANDLE_HPP

#include "legacy_matrix_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace vbcsr::detail {

template <typename T>
class BoundArena {
public:
    BoundArena() = default;
    explicit BoundArena(BlockArena<T>& storage) : storage_(&storage) {}

    void bind(BlockArena<T>& storage) {
        storage_ = &storage;
    }

    T* get_ptr(uint64_t handle) const {
        return storage_->get_ptr(handle);
    }

    uint64_t allocate(size_t size) {
        return storage_->allocate(size);
    }

    void free(uint64_t handle, size_t size) {
        storage_->free(handle, size);
    }

    void clear() {
        storage_->clear();
    }

    void reserve(unsigned long long total_elements) {
        storage_->reserve(total_elements);
    }

    BoundArena& operator=(BlockArena<T>&& other) {
        *storage_ = std::move(other);
        return *this;
    }

    operator BlockArena<T>&() {
        return *storage_;
    }

    operator const BlockArena<T>&() const {
        return *storage_;
    }

private:
    BlockArena<T>* storage_ = nullptr;
};

template <typename T>
class BoundVector {
public:
    BoundVector() = default;
    explicit BoundVector(std::vector<T>& storage) : storage_(&storage) {}

    void bind(std::vector<T>& storage) {
        storage_ = &storage;
    }

    size_t size() const {
        return storage_->size();
    }

    bool empty() const {
        return storage_->empty();
    }

    void clear() {
        storage_->clear();
    }

    void reserve(size_t n) {
        storage_->reserve(n);
    }

    void resize(size_t n) {
        storage_->resize(n);
    }

    T* data() {
        return storage_->data();
    }

    const T* data() const {
        return storage_->data();
    }

    auto begin() {
        return storage_->begin();
    }

    auto end() {
        return storage_->end();
    }

    auto begin() const {
        return storage_->begin();
    }

    auto end() const {
        return storage_->end();
    }

    T& operator[](size_t idx) {
        return (*storage_)[idx];
    }

    const T& operator[](size_t idx) const {
        return (*storage_)[idx];
    }

    BoundVector& operator=(std::vector<T>&& other) {
        *storage_ = std::move(other);
        return *this;
    }

    BoundVector& operator=(const std::vector<T>& other) {
        *storage_ = other;
        return *this;
    }

    friend bool operator==(const BoundVector& lhs, const BoundVector& rhs) {
        return *lhs.storage_ == *rhs.storage_;
    }

    friend bool operator!=(const BoundVector& lhs, const BoundVector& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator==(const BoundVector& lhs, const std::vector<T>& rhs) {
        return *lhs.storage_ == rhs;
    }

    friend bool operator==(const std::vector<T>& lhs, const BoundVector& rhs) {
        return lhs == *rhs.storage_;
    }

    friend bool operator!=(const BoundVector& lhs, const std::vector<T>& rhs) {
        return !(lhs == rhs);
    }

    friend bool operator!=(const std::vector<T>& lhs, const BoundVector& rhs) {
        return !(lhs == rhs);
    }

    operator std::vector<T>&() {
        return *storage_;
    }

    operator const std::vector<T>&() const {
        return *storage_;
    }

private:
    std::vector<T>* storage_ = nullptr;
};

template <typename T>
struct LegacyBackendRef {
    LegacyMatrixBackend<T>* storage = nullptr;
};

template <typename T>
struct CSRMatrixBackend {
    std::vector<uint64_t> blk_handles;
    std::vector<size_t> blk_sizes;
    BlockArena<T> arena;

    size_t local_scalar_nnz() const {
        return blk_handles.size();
    }
};

template <typename T>
struct CSRBackendRef {
    CSRMatrixBackend<T>* storage = nullptr;
};

template <typename T>
struct BSRMatrixBackend {
    int block_size = 0;
    std::vector<uint64_t> blk_handles;
    std::vector<size_t> blk_sizes;
    BlockArena<T> arena;

    size_t local_scalar_nnz() const {
        return blk_handles.size() * static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
    }
};

template <typename T>
struct BSRBackendRef {
    BSRMatrixBackend<T>* storage = nullptr;
};

template <typename T, typename Kernel>
struct VBCSRMatrixBackend {
    bool shape_registry_enabled = false;
};

template <typename T, typename Kernel>
struct VBCSRBackendRef {
    VBCSRMatrixBackend<T, Kernel>* storage = nullptr;
};

template <typename T, typename Kernel>
using MatrixBackendHandle = std::variant<
    std::monostate,
    LegacyBackendRef<T>,
    CSRBackendRef<T>,
    BSRBackendRef<T>,
    VBCSRBackendRef<T, Kernel>>;

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_legacy_backend_handle(LegacyMatrixBackend<T>& storage) {
    return LegacyBackendRef<T>{&storage};
}

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_csr_backend_handle(CSRMatrixBackend<T>& storage) {
    return CSRBackendRef<T>{&storage};
}

template <typename T, typename Kernel>
MatrixBackendHandle<T, Kernel> make_bsr_backend_handle(BSRMatrixBackend<T>& storage) {
    return BSRBackendRef<T>{&storage};
}

template <typename T, typename Kernel>
LegacyMatrixBackend<T>& require_legacy_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<LegacyBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the legacy storage path");
    }
    return *ref->storage;
}

template <typename T, typename Kernel>
const LegacyMatrixBackend<T>& require_legacy_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<LegacyBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the legacy storage path");
    }
    return *ref->storage;
}

template <typename T, typename Kernel>
CSRMatrixBackend<T>& require_csr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<CSRBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the CSR storage path");
    }
    return *ref->storage;
}

template <typename T, typename Kernel>
const CSRMatrixBackend<T>& require_csr_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<CSRBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the CSR storage path");
    }
    return *ref->storage;
}

template <typename T, typename Kernel>
BSRMatrixBackend<T>& require_bsr_backend(MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<BSRBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the BSR storage path");
    }
    return *ref->storage;
}

template <typename T, typename Kernel>
const BSRMatrixBackend<T>& require_bsr_backend(const MatrixBackendHandle<T, Kernel>& handle) {
    auto* ref = std::get_if<BSRBackendRef<T>>(&handle);
    if (ref == nullptr || ref->storage == nullptr) {
        throw std::logic_error("Active backend is not the BSR storage path");
    }
    return *ref->storage;
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BACKEND_HANDLE_HPP
