#include "single-flight.hpp"

#include <atomic>
#include <algorithm>
#include <barrier>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

template <typename T>
concept CanCreateNode = requires { typename Node<T>; };

static_assert(CanCreateNode<int>);
static_assert(!CanCreateNode<std::unique_ptr<int>>);

struct CopyFailure {
    CopyFailure() = default;
    CopyFailure(const CopyFailure&) { throw std::runtime_error("copy failed"); }
    CopyFailure& operator=(const CopyFailure&) = delete;
};

static_assert(CanCreateNode<CopyFailure>);

void test_nodes() {
    constexpr int count = 16;

    {
        Node<int> node;
        assert(!node.check());
        std::atomic<int> calls{0};
        std::barrier start(count);
        std::vector<std::thread> threads;
        for (int i = 0; i < count; ++i) {
            threads.emplace_back([&] {
                start.arrive_and_wait();
                assert(node.call([&] { ++calls; return 42; }) == 42);
            });
        }
        for (auto& thread : threads) thread.join();
        assert(calls == 1);
        assert(node.call([] { return 99; }) == 42);
    }

    {
        Node<int> node;
        std::atomic<int> calls{0};
        std::barrier start(count);
        std::vector<std::exception_ptr> errors(count);
        std::vector<std::thread> threads;
        for (int i = 0; i < count; ++i) {
            threads.emplace_back([&, i] {
                start.arrive_and_wait();
                try {
                    node.call([&]() -> int {
                        ++calls;
                        throw std::runtime_error("computation failed");
                    });
                    assert(false);
                } catch (const std::runtime_error&) {
                    errors[i] = std::current_exception();
                }
            });
        }
        for (auto& thread : threads) thread.join();
        assert(calls == count);
        for (const auto& error : errors) assert(error);
        try {
            node.check();
            assert(false);
        } catch (const std::runtime_error&) {
            const auto last = std::current_exception();
            assert(std::find(errors.begin(), errors.end(), last) != errors.end());
        }
        assert(node.call([] { return 99; }) == 99);
        assert(node.check() == 99);
        assert(node.call([] { return 0; }) == 99);
    }

    {
        Node<CopyFailure> node;
        int calls = 0;
        for (int i = 0; i < 2; ++i) {
            try {
                node.call([&] { ++calls; return CopyFailure{}; });
                assert(false);
            } catch (const std::runtime_error&) {
                const auto failure = std::current_exception();
                try {
                    node.check();
                    assert(false);
                } catch (const std::runtime_error&) {
                    assert(std::current_exception() == failure);
                }
            }
        }
        assert(calls == 2);
    }
}

void test_cache_hits_and_misses() {
    Cache<std::string, int> cache;
    int calls = 0;
    const std::string key = "answer";

    assert(!cache.check(key));
    assert(cache.store.empty());
    assert(calls == 0);
    assert(cache.call(key, [&] { ++calls; return 42; }) == 42);
    assert(cache.call(std::string{"answer"}, [&] { ++calls; return 99; }) == 42);
    assert(cache.check(key) == 42);
    assert(calls == 1);
    assert(cache.call("other", [&] { ++calls; return 7; }) == 7);
    assert(calls == 2);
    assert(!cache.check("missing"));
    assert(calls == 2);
    assert(cache.store.size() == 2);
}

void test_cache_single_flight() {
    constexpr int count = 16;
    Cache<int, int> cache;
    std::atomic<int> calls{0};
    std::barrier start(count);
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future().share();
    std::vector<std::future<int>> results;

    for (int i = 0; i < count; ++i) {
        results.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return cache.call(1, [&] {
                if (++calls == 1) entered.set_value();
                released.wait();
                return 42;
            });
        }));
    }

    entered.get_future().wait();
    release.set_value();
    for (auto& result : results) assert(result.get() == 42);
    assert(calls == 1);
    assert(cache.store.size() == 1);
}

void test_cache_retry_failures() {
    constexpr int count = 16;
    Cache<int, int> cache;
    std::atomic<int> calls{0};
    std::barrier start(count);
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future().share();
    std::vector<std::future<std::exception_ptr>> results;
    for (int i = 0; i < count; ++i) {
        results.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            try {
                cache.call(1, [&]() -> int {
                    if (++calls == 1) entered.set_value();
                    released.wait();
                    throw std::runtime_error("computation failed");
                });
            } catch (const std::runtime_error& error) {
                assert(std::string{error.what()} == "computation failed");
                return std::current_exception();
            }
            return std::exception_ptr{};
        }));
    }
    entered.get_future().wait();
    release.set_value();
    std::vector<std::exception_ptr> errors;
    for (auto& result : results) {
        errors.push_back(result.get());
        assert(errors.back());
    }
    assert(calls == count);
    try {
        cache.check(1);
        assert(false);
    } catch (const std::runtime_error&) {
        const auto last = std::current_exception();
        assert(std::find(errors.begin(), errors.end(), last) != errors.end());
    }
    assert(calls == count); // Checking an error never retries the computation.

    std::barrier retry_start(count);
    std::vector<std::future<int>> retries;
    for (int i = 0; i < count; ++i) {
        retries.push_back(std::async(std::launch::async, [&] {
            retry_start.arrive_and_wait();
            return cache.call(1, [&] { ++calls; return 42; });
        }));
    }
    for (auto& retry : retries) assert(retry.get() == 42);
    assert(calls == count + 1);
    assert(cache.check(1) == 42); // Success clears the previous error.
    assert(cache.call(1, [] { return 99; }) == 42);
    assert(cache.call(2, [] { return 7; }) == 7);
}

void test_independent_keys() {
    using namespace std::chrono_literals;
    Cache<int, int> cache;
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future().share();
    auto slow = std::async(std::launch::async, [&] {
        return cache.call(1, [&] {
            entered.set_value();
            released.wait();
            return 10;
        });
    });
    entered.get_future().wait();

    auto fast = std::async(std::launch::async, [&] {
        return cache.call(2, [] { return 20; });
    });
    // Capture readiness before releasing the first computation. This detects
    // a map lock held across the callback without leaving a blocked worker.
    const bool independent = fast.wait_for(5s) == std::future_status::ready;
    release.set_value();
    assert(slow.get() == 10);
    assert(fast.get() == 20);
    assert(independent);
}

void test_check_waits_for_failure_and_retry() {
    using namespace std::chrono_literals;
    Cache<int, int> cache;

    // First observe a failing computation, then observe its successful retry.
    for (bool fail : {true, false}) {
        std::promise<void> entered;
        std::promise<void> release;
        auto released = release.get_future().share();
        auto producer = std::async(std::launch::async, [&]() -> std::exception_ptr {
            try {
                assert(cache.call(1, [&] {
                    entered.set_value();
                    released.wait();
                    if (fail) throw std::runtime_error("temporary failure");
                    return 42;
                }) == 42);
                return {};
            } catch (const std::runtime_error&) {
                return std::current_exception();
            }
        });
        entered.get_future().wait();
        std::promise<void> checker_started;
        auto checker = std::async(std::launch::async, [&] {
            checker_started.set_value();
            return cache.check(1);
        });
        checker_started.get_future().wait();
        const bool waited = checker.wait_for(50ms) == std::future_status::timeout;
        release.set_value();
        const auto failure = producer.get();
        if (fail) {
            assert(failure);
            try {
                checker.get();
                assert(false);
            } catch (const std::runtime_error&) {
                assert(std::current_exception() == failure);
            }
        } else {
            assert(!failure);
            assert(checker.get() == 42);
        }
        assert(waited);
    }
    assert(cache.check(1) == 42);
}

void test_check_in_flight_and_rehash() {
    using namespace std::chrono_literals;
    Cache<int, int> cache;
    std::promise<void> entered;
    std::promise<void> release;
    auto released = release.get_future().share();
    auto producer = std::async(std::launch::async, [&] {
        return cache.call(0, [&] {
            entered.set_value();
            released.wait();
            return 42;
        });
    });
    entered.get_future().wait();
    std::promise<void> checker_started;
    auto checker = std::async(std::launch::async, [&] {
        checker_started.set_value();
        return cache.check(0);
    });
    checker_started.get_future().wait();
    auto inserter = std::async(std::launch::async, [&] {
        for (int key = 1; key <= 1000; ++key) {
            assert(cache.call(key, [key] { return key; }) == key);
        }
    });
    // Insertions must finish even while a check may be waiting on a node.
    const bool independent = inserter.wait_for(5s) == std::future_status::ready;
    release.set_value();
    assert(producer.get() == 42);
    assert(checker.get() == 42);
    inserter.get();
    assert(independent);
    assert(cache.store.size() == 1001);
    assert(cache.check(0) == 42);
}

struct RvalueCallable {
    explicit RvalueCallable(int value) : value(std::make_unique<int>(value)) {}
    std::unique_ptr<int> value;

    int operator()(std::unique_ptr<int> argument) && {
        return *value + *argument;
    }
};

struct Value {
    explicit Value(int number) : number(number) {}
    int add(int amount) const { return number + amount; }
    int number;
};

void test_forwarding_and_value_types() {
    Cache<int, int> cache;
    assert(cache.call(1, RvalueCallable{40}, std::make_unique<int>(2)) == 42);
    auto unused = std::make_unique<int>(7);
    assert(cache.call(1, RvalueCallable{0}, std::move(unused)) == 42);
    assert(unused); // Cache hits must not consume forwarded arguments.
    assert(!cache.check(2));

    // An existing empty node must remain empty when checked.
    cache.store.try_emplace(3);
    assert(!cache.check(3));
    assert(cache.call(3, RvalueCallable{40}, std::make_unique<int>(2)) == 42);
    assert(cache.check(3) == 42);
    Value value{40};
    assert(cache.call(4, &Value::add, std::cref(value), 2) == 42);

    Cache<int, Value> values;
    assert(values.call(1, [] { return Value{42}; }).number == 42);
    assert(values.check(1)->number == 42);
    Cache<int, std::optional<int>> optional_values;
    assert(!optional_values.call(1, [] { return std::optional<int>{}; }));
    const auto hit = optional_values.check(1);
    assert(hit.has_value() && !hit->has_value());
}

struct CollidingHash {
    std::size_t operator()(int) const { return 0; }
};

void test_hash_collisions_and_nested_calls() {
    Cache<int, int, CollidingHash> cache;
    assert(cache.call(1, [&] {
        return cache.call(2, [] { return 40; }) + 2;
    }) == 42);
    assert(cache.check(1) == 42);
    assert(cache.check(2) == 40);
}

void test_cache_copy_failure() {
    Cache<int, CopyFailure> cache;
    int calls = 0;
    for (int i = 0; i < 2; ++i) {
        try {
            cache.call(1, [&] { ++calls; return CopyFailure{}; });
            assert(false);
        } catch (const std::runtime_error&) {
            const auto failure = std::current_exception();
            try {
                cache.check(1);
                assert(false);
            } catch (const std::runtime_error&) {
                assert(std::current_exception() == failure);
            }
        }
    }
    assert(calls == 2);
}

int main() {
    test_nodes();
    test_cache_hits_and_misses();
    test_cache_single_flight();
    test_cache_retry_failures();
    test_independent_keys();
    test_check_waits_for_failure_and_retry();
    test_check_in_flight_and_rehash();
    test_forwarding_and_value_types();
    test_hash_collisions_and_nested_calls();
    test_cache_copy_failure();
    std::cout << "All single-flight tests passed\n";
}
