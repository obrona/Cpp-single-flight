#include <functional>
#include <optional>
#include <mutex>
#include <utility>
#include <condition_variable>

// trying to implement single flight cache.

template <typename V>
struct Node {
    std::mutex mut;
    std::condition_variable cv;
    bool is_processing = false;
    std::optional<V> res;

    template <typename F, typename... Args>
    V check(F&& f, Args&&... args) {
        std::unique_lock lk{mut};
        
        if (res) {
            return res.value();
        } 
        
        else if (is_processing) {
            cv.wait(lk, [this] () { return res.has_value(); });
            return res.value();
        }

        else {
            is_processing = true;
            lk.unlock();
            
            V val = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
            
            lk.lock();
            is_processing = false;
            res = val;
            cv.notify_all();
            
            return val;
        }
    }
};
