#pragma once
#include <functional>
#include <optional>
#include <mutex>
#include <utility>
#include <condition_variable>
#include <concepts>
#include <exception>
#include <unordered_map>
#include <shared_mutex>

#include "rw-lock.hpp"

// Successful results are cached. check() observes the latest outcome, while
// call() retries a failed computation instead of rethrowing its old exception.
// A callback must not wait on its own in-progress node, directly or indirectly.

template <std::copy_constructible V>
struct Node {
    std::mutex mut;
    std::condition_variable cv;
    bool is_processing = false;
    std::optional<V> res;
    std::exception_ptr error;

    // Wait for any active computation, then return its result or rethrow its
    // error. An empty node returns nullopt; checking never starts computation.
    std::optional<V> check() {
        std::unique_lock lk{mut};
        cv.wait(lk, [this] { return !is_processing; });

        if (error) {
            std::rethrow_exception(error);
        }

        return res;
    }

    // Reuse a successful result, otherwise run the supplied function. Waiting
    // callers retry serially on failure until one produces a cached result.
    template <typename F, typename... Args>
    V call(F&& f, Args&&... args) {
        std::unique_lock lk{mut};
        cv.wait(lk, [this] { return !is_processing; });

        if (res) {
            return res.value();
        }

        is_processing = true;
        lk.unlock();

        try {
            V val = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
            lk.lock();
            res.emplace(val);
            error = nullptr;
        } catch (...) {
            auto failure = std::current_exception();
            if (!lk.owns_lock()) {
                lk.lock();
            }
            error = failure;
        }

        is_processing = false;
        cv.notify_all();

        if (error) {
            std::rethrow_exception(error);
        }
        return res.value();
    }
};

template <typename K, std::copy_constructible V, typename Hash = std::hash<K>>
struct Cache {
    // Nodes must remain alive while callers are active, including across rehash.
    // External access to store requires synchronization with rwlock.
    std::unordered_map<K, Node<V>, Hash> store;
    RWLock rwlock;

    std::optional<V> check(const K& key) {
        std::shared_lock lk{rwlock};
        auto it = store.find(key);
        if (it == store.end()) return std::nullopt;
        auto& node = it->second;
        lk.unlock();
        return node.check();
    }

    template <typename F, typename... Args>
    V call(K key, F&& f, Args&&... args) {
        {
            std::shared_lock lk{rwlock};
            auto it = store.find(key);
            if (it != store.end()) {
                auto& node = it->second;
                lk.unlock();
                return node.call(std::forward<F>(f), std::forward<Args>(args)...);
            }
        }

        std::unique_lock lk{rwlock};
        auto& node = store.try_emplace(std::move(key)).first->second;
        lk.unlock();
        return node.call(std::forward<F>(f), std::forward<Args>(args)...);
    }
};

