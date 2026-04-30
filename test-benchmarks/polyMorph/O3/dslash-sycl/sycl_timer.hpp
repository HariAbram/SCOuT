#pragma once
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef SYCL_TIMER_ONLY_AGG
#define SYCL_TIMER_ONLY_AGG 1
#endif

namespace sycl_timer_detail {

inline const char* out_path() {
  if (const char* p = std::getenv("SYCL_TIMER_OUT")) return p;
  return "sycl_timing.csv";
}

inline std::mutex& mtx() { static std::mutex m; return m; }

struct Agg { double sum_ms = 0.0; long long count = 0; };

inline std::unordered_map<std::string, Agg>& agg_map() {
  static std::unordered_map<std::string, Agg> m;
  return m;
}

inline void record_agg(const std::string& label, double ms) {
  std::lock_guard<std::mutex> lock(mtx());
  auto& a = agg_map()[label];
  a.sum_ms += ms;
  a.count  += 1;
}

inline void dump_aggregates() {
  std::lock_guard<std::mutex> lock(mtx());
  if (agg_map().empty()) return;

  // CSV: write sum and avg
  if (std::FILE* f = std::fopen(out_path(), "a")) {
    for (const auto& kv : agg_map()) {
      const auto& label = kv.first;
      const auto& a = kv.second;
      const double avg = (a.count>0) ? (a.sum_ms / double(a.count)) : 0.0;
      std::fprintf(f, "%s(sum),%.6f,%lld\n", label.c_str(), a.sum_ms/1000.0, a.count);
      std::fprintf(f, "%s(avg),%.6f,%lld\n", label.c_str(), avg/1000.0, a.count);
    }
    std::fclose(f);
  }

  // Terminal: only aggregates
  for (const auto& kv : agg_map()) {
    const auto& label = kv.first;
    const auto& a = kv.second;
    const double avg = (a.count>0) ? (a.sum_ms / double(a.count)) : 0.0;
    std::fprintf(stdout, "[SYCL][sum] %s: %.6f s over %lld iters\n",
                 label.c_str(), a.sum_ms/1000.0, a.count);
    std::fprintf(stdout, "[SYCL][avg] %s: %.6f s over %lld iters\n",
                 label.c_str(), avg/1000.0, a.count);
  }
  std::fflush(stdout);
}

struct AggDumper {
  AggDumper() { std::atexit(&dump_aggregates); }
  ~AggDumper() { dump_aggregates(); }
};
static AggDumper _sycl_timer_agg_dumper_guard{};

inline void dump_now() { dump_aggregates(); }

inline double duration_ms_from_event(const sycl::event& e) {
  try {
    auto s = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return double(t - s) / 1.0e6;
  } catch (...) { return -1.0; }
}

template <class F>
inline sycl::event time_expr_agg(const char* label, F&& f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  sycl::event evt = std::forward<F>(f)();
  { std::vector<sycl::event> __ev{evt}; sycl::event::wait(__ev); }
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = duration_ms_from_event(evt);
  if (ms < 0.0) ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  record_agg(label, ms);
  return evt;
}

inline void record_event_ms(const char* label, const sycl::event& evt) {
  { std::vector<sycl::event> __ev{evt}; sycl::event::wait(__ev); }
  double ms = duration_ms_from_event(evt);
  if (ms < 0.0) return; // profiling unavailable; skip (keeps behavior minimal)
  record_agg(label, ms);
}

} // namespace sycl_timer_detail

#define SYCL_TIME_EVT(LABEL, EVT) (::sycl_timer_detail::record_event_ms((LABEL),(EVT)))
#define SYCL_TIMER_DUMP() ::sycl_timer_detail::dump_now()


#define SYCL_TIME_AGG(LABEL, EXPR) \
  (::sycl_timer_detail::time_expr_agg((LABEL), [&](){ return (EXPR); }))

// For safety: SYCL_TIME behaves identically (aggregate-only)
#ifndef SYCL_TIME
#define SYCL_TIME(LABEL, EXPR) SYCL_TIME_AGG(LABEL, EXPR)
#endif

