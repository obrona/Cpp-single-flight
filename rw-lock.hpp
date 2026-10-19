#pragma once
#include <mutex>
#include <condition_variable>

struct RWLock {
    std::mutex m;
    std::condition_variable readers_cv;
    std::condition_variable writers_cv;
    int  active_readers  = 0;
    int  waiting_writers = 0;
    bool writer_active  = false;

    void lock_shared() {
        std::unique_lock lk{m};
        readers_cv.wait(lk, [this] () {
            return waiting_writers == 0 && !writer_active;
        });
        active_readers++;
    }

    void unlock_shared() {
        std::unique_lock lk{m};
        if (--active_readers == 0 && waiting_writers > 0) {
            writers_cv.notify_one();
        }
    }

    void lock() {
        std::unique_lock lk{m};
        waiting_writers++;
        writers_cv.wait(lk, [this] () {
            return active_readers == 0 && writer_active == 0;
        });
        waiting_writers--;
        writer_active = true;
    }

    void unlock() {
        std::unique_lock lk{m};
        writer_active = false;

        if (waiting_writers > 0) {
            writers_cv.notify_one();
        } else {
            readers_cv.notify_all();
        }
    }
};