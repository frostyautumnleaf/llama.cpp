#include "llama-expert-stream.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-mmap.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#ifndef MADV_PAGEOUT
#define MADV_PAGEOUT 21
#endif
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

// ---------------------------------------------------------------------------
// the operating-system layer
//
//   Linux    readahead + MADV_POPULATE_READ, MADV_PAGEOUT / MADV_DONTNEED + fadvise, mincore
//   macOS    MADV_WILLNEED + touching the pages, MADV_DONTNEED (a soft budget), mincore
//   Windows  PrefetchVirtualMemory + touching the pages, VirtualUnlock to trim the working set
//            (a soft budget), no residency ground truth
// ---------------------------------------------------------------------------

namespace expert_os {

static size_t page_size() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (size_t) si.dwPageSize;
#else
    return (size_t) sysconf(_SC_PAGESIZE);
#endif
}

static int open_ro(const std::string & path) {
#if defined(_WIN32)
    return _open(path.c_str(), _O_RDONLY | _O_BINARY);
#else
    return open(path.c_str(), O_RDONLY | O_NOFOLLOW);
#endif
}

static void close_fd(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    close(fd);
#endif
}

// fraction of [a, a+len) resident in memory, or -1 when the platform cannot tell
static double residency(void * a, size_t len, size_t page) {
#if defined(_WIN32)
    (void) a; (void) len; (void) page;
    return -1.0;
#else
    std::vector<unsigned char> vec(len / page);
#if defined(__linux__)
    if (mincore(a, len, vec.data()) != 0) {
        return -1.0;
    }
#else
    if (mincore(a, len, (char *) vec.data()) != 0) {
        return -1.0;
    }
#endif
    size_t present = 0;
    for (unsigned char c : vec) {
        present += c & 1;
    }
    return vec.empty() ? 1.0 : (double) present / (double) vec.size();
#endif
}

// bring [a, a+len) of a file range into memory and map it
static void fetch(int fd, uint64_t off, size_t bytes, uint8_t * a, size_t len, size_t page) {
#if defined(__linux__)
    (void) page;
    if (readahead(fd, (off64_t) off, bytes) != 0) {
        posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_WILLNEED);
    }
    madvise(a, len, MADV_POPULATE_READ);
#else
    (void) fd; (void) off; (void) bytes;
#if defined(_WIN32)
    WIN32_MEMORY_RANGE_ENTRY entry{ a, len };
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#elif defined(MADV_WILLNEED)
    madvise(a, len, MADV_WILLNEED);
#endif
    volatile uint8_t sink = 0;
    for (size_t o = 0; o < len; o += page) {
        sink ^= a[o];  // fault the pages in
    }
    (void) sink;
#endif
}

// let the operating system reclaim a file range; [a, a+len) is the page-aligned part inside it
static void drop(int fd, uint64_t off, size_t bytes, uint8_t * a, size_t len, bool pageout) {
#if defined(__linux__)
    if (len) {
        madvise(a, len, pageout ? MADV_PAGEOUT : MADV_DONTNEED);
    }
    posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED);
#elif defined(_WIN32)
    (void) fd; (void) off; (void) bytes; (void) pageout;
    if (len) {
        VirtualUnlock(a, len);  // on unlocked pages this trims the working set
    }
#else
    (void) fd; (void) off; (void) bytes; (void) pageout;
#if defined(MADV_DONTNEED)
    if (len) {
        madvise(a, len, MADV_DONTNEED);
    }
#endif
#endif
}

struct io_counters {
    uint64_t read_bytes  = 0;
    long     major_faults = 0;
    long     minor_faults = 0;
};

static io_counters get_io_counters() {
    io_counters c;
#if defined(_WIN32)
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) {
        c.read_bytes = io.ReadTransferCount;
    }
#elif defined(__linux__)
    std::ifstream f("/proc/self/io");  // Linux only; zero elsewhere
    std::string k;
    uint64_t    v;
    while (f >> k >> v) {
        if (k == "read_bytes:") {
            c.read_bytes = v;
        }
    }
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    c.major_faults = ru.ru_majflt;
    c.minor_faults = ru.ru_minflt;
#endif
    return c;
}

}  // namespace expert_os

// ---------------------------------------------------------------------------
// a worker pool: whole experts are read on several threads at once
// ---------------------------------------------------------------------------

namespace {

class io_pool {
public:
    explicit io_pool(int n) {
        for (int i = 0; i < n; i++) {
            workers.emplace_back([this] { loop(); });
        }
    }

    ~io_pool() {
        {
            std::lock_guard<std::mutex> g(m);
            stop = true;
        }
        cv.notify_all();
        for (auto & t : workers) {
            t.join();
        }
    }

    void submit(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> g(m);
            q.push_back(std::move(job));
            pending++;
        }
        cv.notify_one();
    }

    void drain() {
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [this] { return pending == 0; });
    }

    // run every job and wait for all of them
    void run(std::vector<std::function<void()>> & jobs) {
        if (jobs.empty()) {
            return;
        }
        {
            std::lock_guard<std::mutex> g(m);
            for (auto & j : jobs) {
                q.push_back(std::move(j));
            }
            pending += jobs.size();
        }
        cv.notify_all();
        std::unique_lock<std::mutex> l(m);
        done_cv.wait(l, [this] { return pending == 0; });
        jobs.clear();
    }

private:
    void loop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> l(m);
                cv.wait(l, [this] { return stop || !q.empty(); });
                if (stop && q.empty()) {
                    return;
                }
                job = std::move(q.front());
                q.pop_front();
            }
            job();
            {
                std::lock_guard<std::mutex> g(m);
                if (--pending == 0) {
                    done_cv.notify_all();
                }
            }
        }
    }

    std::vector<std::thread>          workers;
    std::list<std::function<void()>>  q;
    std::mutex                        m;
    std::condition_variable           cv, done_cv;
    size_t                            pending = 0;
    bool                              stop    = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// the expert map of the model
// ---------------------------------------------------------------------------

void llama_expert_pool_build(const llama_model & model, llama_model_loader & ml, llama_expert_pool & pool) {
    pool = llama_expert_pool{};

    const int64_t n_layer = model.hparams.n_layer();
    if (n_layer <= 0 || (int64_t) model.layers.size() < n_layer) {
        pool.blocked = "the model has no layers";
        return;
    }

    // the file paths, and the memory window each file is mapped to
    std::vector<std::pair<const void *, size_t>> windows;
    for (uint32_t f = 0; f < ml.files.size(); f++) {
        pool.files.push_back(ml.files[f]->path());
        const void * addr = f < ml.mappings.size() && ml.mappings[f] ? ml.mappings[f]->addr() : nullptr;
        const size_t size = f < ml.mappings.size() && ml.mappings[f] ? ml.mappings[f]->size() : 0;
        pool.base.push_back(addr);
        windows.emplace_back(addr, size);
    }

    pool.layers.resize((size_t) n_layer);

    // the file offset of a pointer inside a mapping, -1 when it is not file-backed
    auto file_offset_of = [&](const void * p, int & file) -> int64_t {
        for (size_t f = 0; f < windows.size(); f++) {
            const char * beg = (const char *) windows[f].first;
            if (beg == nullptr) {
                continue;
            }
            const char * end = beg + windows[f].second;
            if ((const char *) p >= beg && (const char *) p < end) {
                file = (int) f;
                return (int64_t) ((const char *) p - beg);
            }
        }
        return -1;
    };

    for (int64_t il = 0; il < n_layer; il++) {
        const auto & ll = model.layers[il];

        // the tensors whose rows are the routed experts; a shared expert is used by every token,
        // so it is left to the page cache like any other dense weight
        const ggml_tensor * w[3];
        int n_w = 0;
        if (ll.ffn_gate_up_exps) {
            w[n_w++] = ll.ffn_gate_up_exps;
        } else {
            if (ll.ffn_up_exps)   w[n_w++] = ll.ffn_up_exps;
            if (ll.ffn_gate_exps) w[n_w++] = ll.ffn_gate_exps;
        }
        if (ll.ffn_down_exps) {
            w[n_w++] = ll.ffn_down_exps;
        }
        if (n_w == 0) {
            continue;
        }

        llama_expert_pool::layer out;
        for (int i = 0; i < n_w; i++) {
            if (ggml_n_dims(w[i]) != 3 || !ggml_is_contiguous(w[i])) {
                pool.blocked = "the expert tensor " + std::string(w[i]->name) + " is not a plain 3-D tensor";
                return;
            }
            int file = -1;
            const int64_t off = file_offset_of(w[i]->data, file);
            if (off < 0) {
                pool.blocked = "the expert tensor " + std::string(w[i]->name) +
                               " is not a window of the model file (was it copied, repacked or moved to a GPU?)";
                return;
            }
            llama_expert_pool::ref r;
            r.w        = w[i];
            r.file     = file;
            r.file_off = (uint64_t) off;
            out.expert_bytes += (size_t) w[i]->nb[2];
            if (pool.n_expert == 0) {
                pool.n_expert = (int) w[i]->ne[2];
            } else if (pool.n_expert != (int) w[i]->ne[2]) {
                pool.blocked = "the expert count differs between layers";
                return;
            }
            out.refs.push_back(r);
        }
        if (out.refs.empty() || out.expert_bytes == 0) {
            continue;
        }
        pool.pool_bytes += out.expert_bytes * (size_t) pool.n_expert;
        out.refs.shrink_to_fit();
        pool.layers[il] = std::move(out);
    }

    if (pool.n_expert <= 0 || pool.pool_bytes == 0) {
        pool = llama_expert_pool{};
        pool.blocked = "the model has no routed experts";
        return;
    }

    for (const auto & path : pool.files) {
        if (path.empty()) {
            pool.blocked = "the model was not loaded from files, so there is nothing to stream";
            pool.layers.clear();
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// the engine
// ---------------------------------------------------------------------------

namespace {

struct slice {
    const uint8_t * ptr      = nullptr;  // inside llama.cpp's mapping
    size_t          bytes    = 0;
    int             fd       = -1;       // our own descriptor of the same file
    uint64_t        file_off = 0;
};

struct expert_slice_set {
    std::vector<slice> slices;
    size_t             bytes = 0;
};

struct counters {
    uint64_t accesses = 0, hits = 0, misses = 0, evictions = 0, rejected = 0;
    uint64_t fetched_bytes = 0;
    uint64_t resident_checked = 0, resident_true = 0;
    double   fetch_seconds = 0, evict_seconds = 0, callback_seconds = 0;
};

using clk = std::chrono::steady_clock;

static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

}  // namespace

struct llama_expert_stream::impl {
    const llama_model &            model;
    llama_expert_stream_params     p;
    const llama_expert_pool &      pool;
    const size_t                   page = expert_os::page_size();

    std::vector<int> fds;                       // one read-only descriptor per model file
    std::string      error;
    bool             started = false;

    // decayed usage counts: value and the step it was last updated at
    std::unordered_map<int64_t, std::pair<double, uint64_t>> freq;
    std::unordered_map<int64_t, double> heat;   // per-token use counts, saved with heat_out

    // fetched for the layer that asked and not admitted: paged out at the next callback
    std::vector<int64_t> transient;

    std::unordered_map<int64_t, expert_slice_set> experts;  // key = il * n_expert + e
    std::list<int64_t>                            lru;      // resident experts, coldest first
    std::unordered_map<int64_t, std::list<int64_t>::iterator> where;
    size_t resident_bytes = 0;

    uint64_t step       = 0;
    int      last_layer = -1;

    counters st;
    uint64_t disk_start = 0;

    std::unique_ptr<io_pool> fetch_pool;
    std::unique_ptr<io_pool> evictor;

    impl(const llama_model & model_, const llama_expert_stream_params & p_) :
        model(model_), p(p_), pool(model.expert_pool) {
        if (!pool.enabled()) {
            error = pool.blocked.empty() ? "this model has no streamable experts" : pool.blocked;
            return;
        }
        for (size_t f = 0; f < pool.files.size(); f++) {
            const int fd = expert_os::open_ro(pool.files[f]);
            if (fd < 0) {
                error = "cannot reopen " + pool.files[f];
                break;
            }
            fds.push_back(fd);
        }
        // the tensors must still be where the pool says they are: a weight that was copied,
        // repacked or moved to a GPU after the map was read cannot be streamed from the file
        for (const auto & l : pool.layers) {
            for (const auto & r : l.refs) {
                const void * want = r.file < (int) pool.base.size() && pool.base[r.file] ?
                    (const void *) ((const char *) pool.base[r.file] + r.file_off) : nullptr;
                if (r.w->data != want) {
                    error = "the expert tensor " + std::string(r.w->name) +
                            " is no longer where the model file has it, so it cannot be streamed";
                    break;
                }
            }
            if (!error.empty()) {
                break;
            }
        }
        if (!error.empty()) {
            return;
        }
        if (p.io_threads < 1) {
            p.io_threads = 1;
        }
        fetch_pool = std::make_unique<io_pool>(p.io_threads);
        if (p.async_evict) {
            evictor = std::make_unique<io_pool>(1);
        }
        if (p.heat_in && *p.heat_in) {
            std::ifstream f(p.heat_in);
            int64_t k;
            double  h;
            double  total = 0;
            size_t  n     = 0;
            while (f >> k >> h) {
                heat[k] = h;
                total += h;
                n++;
            }
            if (n) {
                // heat holds per-token use counts from an earlier run; seed the admission
                // frequencies with them so a warm start ranks experts the way the last run used
                for (auto & kv : heat) {
                    freq[kv.first] = { kv.second / total * (double) std::max<size_t>(1, n), 0 };
                }
            }
        }
        if (p.cold) {
            evict_all();
        }
        started = true;
        if (p.heat_in && *p.heat_in) {
            const size_t warmed = warm_start();
            LLAMA_LOG_INFO("%s: warm-started %zu experts from %s\n", __func__, warmed, p.heat_in);
        }
        disk_start = expert_os::get_io_counters().read_bytes;
    }

    ~impl() {
        if (p.heat_out && *p.heat_out && !heat.empty()) {
            std::ofstream f(p.heat_out);
            for (auto & kv : heat) {
                f << kv.first << " " << kv.second << "\n";
            }
        }
        for (int fd : fds) {
            expert_os::close_fd(fd);
        }
    }

    // the slices of one expert, built on first use
    const expert_slice_set * expert(int il, int e) {
        const int64_t key = (int64_t) il * pool.n_expert + e;
        auto it = experts.find(key);
        if (it != experts.end()) {
            return it->second.slices.empty() ? nullptr : &it->second;
        }
        if (il < 0 || il >= (int) pool.layers.size() || pool.layers[il].refs.empty()) {
            return nullptr;
        }
        expert_slice_set x;
        for (const auto & r : pool.layers[il].refs) {
            slice s;
            s.ptr      = (const uint8_t *) r.w->data + (size_t) e * r.w->nb[2];
            s.bytes    = (size_t) r.w->nb[2];
            s.fd       = fds[(size_t) r.file];
            s.file_off = r.file_off + (uint64_t) e * r.w->nb[2];
            x.bytes += s.bytes;
            x.slices.push_back(s);
        }
        return &experts.emplace(key, std::move(x)).first->second;
    }

    static void aligned_outer(const uint8_t * ptr, size_t n, size_t page, uint8_t *& a, size_t & len) {
        const uintptr_t s = (uintptr_t) ptr & ~(uintptr_t) (page - 1);
        const uintptr_t e = ((uintptr_t) ptr + n + page - 1) & ~(uintptr_t) (page - 1);
        a   = (uint8_t *) s;
        len = e - s;
    }

    static bool aligned_inner(const uint8_t * ptr, size_t n, size_t page, uint8_t *& a, size_t & len) {
        const uintptr_t s = ((uintptr_t) ptr + page - 1) & ~(uintptr_t) (page - 1);
        const uintptr_t e = ((uintptr_t) ptr + n) & ~(uintptr_t) (page - 1);
        if (e <= s) {
            return false;
        }
        a   = (uint8_t *) s;
        len = e - s;
        return true;
    }

    // fraction of the expert's pages present in the page cache and mapped
    double residency(const expert_slice_set & x) const {
        double present = 0, total = 0;
        for (const slice & s : x.slices) {
            uint8_t * a;
            size_t    len;
            aligned_outer(s.ptr, s.bytes, page, a, len);
            const double r = expert_os::residency(a, len, page);
            if (r < 0) {
                return -1;
            }
            present += r * (double) len;
            total += (double) len;
        }
        return total > 0 ? present / total : 1.0;
    }

    void drop_slice(const slice & s, bool pageout) const {
        uint8_t * a   = nullptr;
        size_t    len = 0;
        if (!aligned_inner(s.ptr, s.bytes, page, a, len)) {
            len = 0;
        }
        expert_os::drop(s.fd, s.file_off, s.bytes, a, len, pageout);
    }

    void evict(const expert_slice_set & x) {
        const auto t0 = clk::now();
        if (evictor) {
            const bool po = p.pageout;
            for (const slice & s : x.slices) {
                evictor->submit([this, s, po] { drop_slice(s, po); });
            }
        } else {
            for (const slice & s : x.slices) {
                drop_slice(s, p.pageout);
            }
        }
        st.evict_seconds += secs(t0, clk::now());
    }

    void fetch_one(const slice & s) const {
        uint8_t * a;
        size_t    len;
        aligned_outer(s.ptr, s.bytes, page, a, len);
        expert_os::fetch(s.fd, s.file_off, s.bytes, a, len, page);
    }

    void touch(int64_t key) {
        auto it = where.find(key);
        if (it == where.end()) {
            return;
        }
        lru.splice(lru.end(), lru, it->second);
    }

    double frequency(int64_t key) const {
        auto it = freq.find(key);
        if (it == freq.end()) {
            return 0.0;
        }
        return it->second.first * std::exp2(-(double) (step - it->second.second) / std::max(1.0, p.half_life));
    }

    void bump(int64_t key, double w) {
        const double f = frequency(key) + w;
        freq[key] = { f, step };
    }

    // TinyLFU-style admission: a missed expert is kept only if it is used more often than the
    // expert it would evict, so a one-off expert cannot push out a hot one
    bool should_admit(int64_t key, const expert_slice_set & x, const std::set<int64_t> & protect) const {
        if (p.policy == LLAMA_EXPERT_POLICY_LRU || resident_bytes + x.bytes <= p.budget_bytes) {
            return true;
        }
        double need = 0, victims = 0;
        for (auto it = lru.begin(); it != lru.end() && resident_bytes + x.bytes - need > p.budget_bytes; ++it) {
            if (protect.count(*it)) {
                continue;
            }
            victims = std::max(victims, frequency(*it));
            need += experts.at(*it).bytes;
        }
        return frequency(key) > victims;
    }

    void admit(int64_t key, const expert_slice_set & x, const std::set<int64_t> & protect) {
        while (resident_bytes + x.bytes > p.budget_bytes && !lru.empty()) {
            // evict the least recently used expert this layer does not need right now
            auto victim = lru.begin();
            while (victim != lru.end() && protect.count(*victim)) {
                ++victim;
            }
            if (victim == lru.end()) {
                break;
            }
            const int64_t vk = *victim;
            evict(experts.at(vk));
            resident_bytes -= experts.at(vk).bytes;
            where.erase(vk);
            lru.erase(victim);
            st.evictions++;
        }
        lru.push_back(key);
        where[key] = std::prev(lru.end());
        resident_bytes += x.bytes;
    }

    void evict_all() {
        for (size_t il = 0; il < pool.layers.size(); il++) {
            if (pool.layers[il].refs.empty()) {
                continue;
            }
            for (int e = 0; e < pool.n_expert; e++) {
                const expert_slice_set * x = expert((int) il, e);
                if (x) {
                    evict(*x);
                }
            }
        }
        if (evictor) {
            evictor->drain();
        }
        lru.clear();
        where.clear();
        resident_bytes = 0;
    }

    // preload the hottest experts until the budget is full
    size_t warm_start() {
        std::vector<std::pair<double, int64_t>> order;
        for (auto & kv : heat) {
            order.emplace_back(kv.second, kv.first);
        }
        std::sort(order.begin(), order.end(), [](auto & a, auto & b) { return a.first > b.first; });

        std::vector<std::function<void()>> jobs;
        size_t n = 0;
        for (auto & o : order) {
            const int il = (int) (o.second / pool.n_expert);
            const int e  = (int) (o.second % pool.n_expert);
            if (il < 0 || il >= (int) pool.layers.size()) {
                continue;
            }
            const expert_slice_set * x = expert(il, e);
            if (!x || where.count(o.second)) {
                continue;
            }
            if (resident_bytes + x->bytes > p.budget_bytes) {
                break;
            }
            lru.push_front(o.second);  // coldest first, so recency takes over quickly
            where[o.second] = lru.begin();
            resident_bytes += x->bytes;
            for (const slice & s : x->slices) {
                jobs.push_back([this, s] { fetch_one(s); });
            }
            n++;
        }
        fetch_pool->run(jobs);
        return n;
    }

    void on_selected(int il, const int32_t * ids, size_t n_ids, int64_t n_tokens) {
        if (il <= last_layer) {
            step++;  // a new ubatch started
        }
        last_layer = il;

        std::set<int64_t>         keys;
        std::map<int64_t, int>    occ;
        for (size_t i = 0; i < n_ids; i++) {
            const int64_t k = (int64_t) il * pool.n_expert + ids[i];
            keys.insert(k);
            occ[k]++;
        }

        // page out what only the previous layer needed
        for (int64_t k : transient) {
            if (!where.count(k) && !keys.count(k)) {
                auto it = experts.find(k);
                if (it != experts.end()) {
                    evict(it->second);
                }
            }
        }
        transient.clear();

        // per token, so a long prefill counts like one token's worth of expert use
        for (auto & kv : occ) {
            bump(kv.first, (double) kv.second / (double) std::max<int64_t>(1, n_tokens));
        }

        std::vector<std::function<void()>> jobs;
        for (int64_t key : keys) {
            const int e = (int) (key % pool.n_expert);
            const expert_slice_set * x = expert(il, e);
            if (!x) {
                continue;
            }
            st.accesses++;
            heat[key] += (double) occ[key] / (double) std::max<int64_t>(1, n_tokens);

            if (p.check_residency) {
                const double r = residency(*x);
                if (r >= 0) {
                    st.resident_checked++;
                    if (r >= 0.99) {
                        st.resident_true++;
                    }
                }
            }

            if (where.count(key)) {
                st.hits++;
                touch(key);
                continue;
            }
            st.misses++;
            if (should_admit(key, *x, keys)) {
                admit(key, *x, keys);
            } else {
                st.rejected++;
                transient.push_back(key);
            }
            if (p.prefetch) {
                for (const slice & s : x->slices) {
                    jobs.push_back([this, s] { fetch_one(s); });
                }
                st.fetched_bytes += x->bytes;
            }
        }

        if (!jobs.empty()) {
            const auto t0 = clk::now();
            fetch_pool->run(jobs);
            st.fetch_seconds += secs(t0, clk::now());
        }
    }

    void fill(llama_expert_stream_stats * out) const {
        *out = llama_expert_stream_stats{};
        out->accesses         = st.accesses;
        out->hits             = st.hits;
        out->misses           = st.misses;
        out->evictions        = st.evictions;
        out->rejected         = st.rejected;
        out->fetched_bytes    = st.fetched_bytes;
        out->resident_checked = st.resident_checked;
        out->resident_true    = st.resident_true;
        out->fetch_seconds    = st.fetch_seconds;
        out->evict_seconds    = st.evict_seconds;
        out->callback_seconds = st.callback_seconds;
        out->resident_bytes   = resident_bytes;
        out->budget_bytes     = p.budget_bytes;
        out->pool_bytes       = pool.pool_bytes;
        out->disk_read_bytes  = expert_os::get_io_counters().read_bytes - disk_start;
        int n_moe             = 0;
        for (const auto & l : pool.layers) {
            if (!l.refs.empty()) {
                n_moe++;
            }
        }
        out->n_layer_moe = n_moe;
        out->n_expert    = pool.n_expert;
    }
};

// ---------------------------------------------------------------------------
// public surface
// ---------------------------------------------------------------------------

llama_expert_stream_params llama_expert_stream_default_params(void) {
    llama_expert_stream_params p;
    p.budget_bytes    = 0;   // streaming off
    p.io_threads      = 8;
    p.policy          = LLAMA_EXPERT_POLICY_ADMIT;
    p.half_life       = 256;
    p.async_evict     = true;
    p.prefetch        = true;
    p.pageout         = false;
    p.check_residency = true;
    p.cold            = true;
    p.heat_in         = nullptr;
    p.heat_out        = nullptr;
    return p;
}

llama_expert_stream::llama_expert_stream(const llama_model & model, const llama_expert_stream_params & params) :
    pimpl(new impl(model, params)) {}

llama_expert_stream::~llama_expert_stream() = default;

bool llama_expert_stream::enabled() const {
    return pimpl->started && pimpl->error.empty();
}

const std::string & llama_expert_stream::error() const {
    return pimpl->error;
}

void llama_expert_stream::on_selected(int il, const int32_t * ids, size_t n_ids, int64_t n_tokens) {
    pimpl->on_selected(il, ids, n_ids, n_tokens);
}

void llama_expert_stream::get_stats(llama_expert_stream_stats * stats) const {
    pimpl->fill(stats);
}

void llama_expert_stream::log_stats() const {
    llama_expert_stream_stats s;
    pimpl->fill(&s);
    LLAMA_LOG_INFO("%s: %d MoE layers x %d experts, pool %.2f GB, budget %.2f GB, resident %.2f GB, "
                   "accesses %llu, hit %.1f%% (page cache %.1f%%), evictions %llu, rejected %llu, "
                   "fetched %.2f GB, disk %.2f GB, fetch %.2f s, evict %.2f s, callback %.2f s\n",
                   __func__, (int) s.n_layer_moe, (int) s.n_expert, s.pool_bytes / 1e9, s.budget_bytes / 1e9,
                   s.resident_bytes / 1e9, (unsigned long long) s.accesses,
                   s.accesses ? 100.0 * s.hits / s.accesses : 0.0,
                   s.resident_checked ? 100.0 * s.resident_true / s.resident_checked : 0.0,
                   (unsigned long long) s.evictions, (unsigned long long) s.rejected,
                   s.fetched_bytes / 1e9, s.disk_read_bytes / 1e9,
                   s.fetch_seconds, s.evict_seconds, s.callback_seconds);
}
