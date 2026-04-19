#ifndef VBCSR_DETAIL_REMOTE_ASSEMBLY_HPP
#define VBCSR_DETAIL_REMOTE_ASSEMBLY_HPP

#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace vbcsr::detail {

template <typename Matrix>
struct RemoteAssemblyState {
    using T = typename Matrix::value_type;

    struct PendingBlock {
        int rows = 0;
        int cols = 0;
        int mode_code = 0;
        std::vector<T> data;
    };

    using RemoteOwnerBlocks = std::map<int, std::map<std::pair<int, int>, PendingBlock>>;
    using RemoteThreadBuffers = std::vector<RemoteOwnerBlocks>;

    struct Registry {
        std::mutex mutex;
        std::map<const Matrix*, RemoteThreadBuffers> buffers;
    };

    static Registry& registry() {
        static Registry instance;
        return instance;
    }

    static RemoteThreadBuffers& buffers_for(const Matrix* matrix, int thread_count) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto& buffers = reg.buffers[matrix];
        if (buffers.empty()) {
            buffers.resize(static_cast<size_t>(thread_count));
        }
        return buffers;
    }

    static void transfer(const Matrix* from, const Matrix* to) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.buffers.find(from);
        if (it == reg.buffers.end()) {
            return;
        }
        reg.buffers[to] = std::move(it->second);
        reg.buffers.erase(it);
    }

    static void clear(const Matrix* matrix) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        reg.buffers.erase(matrix);
    }

    static bool has_pending(const Matrix* matrix) {
        auto& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mutex);
        auto it = reg.buffers.find(matrix);
        if (it == reg.buffers.end()) {
            return false;
        }
        for (const auto& thread_buffers : it->second) {
            for (const auto& owner_entry : thread_buffers) {
                if (!owner_entry.second.empty()) {
                    return true;
                }
            }
        }
        return false;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_REMOTE_ASSEMBLY_HPP
