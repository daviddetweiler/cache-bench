#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#include <x86intrin.h>

#include <pthread.h>

namespace cachebench {
	namespace {
		std::uint64_t start_timed(void)
		{
			unsigned cycles_low, cycles_high;

			asm volatile("CPUID\n\t"
						 "RDTSC\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");

			return (static_cast<std::uint64_t>(cycles_high) << 32) | cycles_low;
		}

		std::uint64_t end_timed(void)
		{
			unsigned cycles_low, cycles_high;

			asm volatile("RDTSCP\n\t"
						 "mov %%edx, %0\n\t"
						 "mov %%eax, %1\n\t"
						 "CPUID\n\t"
						 : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");

			return (static_cast<std::uint64_t>(cycles_high) << 32) | cycles_low;
		}

		using seconds = std::chrono::duration<double>;

		struct bench_stats {
			double bandwidth;
		};

		bench_stats bench_read(std::uint64_t size)
		{
			using type = std::uint64_t;
			std::vector<type> buffer(size / sizeof(type), 1u);
			std::vector<type> decoy(size / sizeof(type), 1u);

			using clock = std::chrono::high_resolution_clock;

			const auto start_time = clock::now();
			const auto start = start_timed();
			volatile std::uint64_t foo {};
			for (std::uint64_t i {}; i < buffer.size(); i += 8) {
				foo = buffer.at(i);
			}

			const auto end = end_timed();
			const auto end_time = clock::now();
			const auto elapsed = std::chrono::duration_cast<seconds>(end_time - start_time);

			const bench_stats stats {size / elapsed.count()};
			std::stringstream stream {};
			stream << stats.bandwidth << " B/s\n";
			std::cout << stream.str();

			return stats;
		}
	}
}

int main(int argc, char** argv)
{
	using namespace cachebench;

	const std::uint32_t thread_count = std::thread::hardware_concurrency();
	std::atomic_uint64_t n_spinning {};

	if (argc != 2)
		return 1;

	const auto type = std::stoi(argv[1]);
	const auto bench = [&type] {
		switch (type) {
		case 0:
			return bench_read;
		}

		std::terminate();
	}();

	std::vector<std::thread> threads(thread_count - 1);
	std::vector<bench_stats> stats(thread_count);
	for (std::int64_t i {}; i < thread_count - 1; ++i) {
		threads.at(i) = std::thread {[&n_spinning, &stat_cell = stats.at(i), thread_count, bench] {
			++n_spinning;
			while (n_spinning != thread_count)
				_mm_pause();

			stat_cell = bench(1'000'000'000);

			--n_spinning;
		}};

		cpu_set_t cpuset {};
		CPU_ZERO(&cpuset);
		CPU_SET(i, &cpuset);
		const auto err = pthread_setaffinity_np(threads.at(i).native_handle(), sizeof(cpuset), &cpuset);
		if (err) {
			std::cout << "Returned " << err << "\n";
			return 1;
		}
	}

	++n_spinning;
	while (n_spinning != thread_count)
		_mm_pause();

	stats.back() = bench(1'000'000'000);

	--n_spinning;

	while (n_spinning)
		_mm_pause();

	double bandwidth {};
	for (const auto& stat_cell : stats) {
		bandwidth += stat_cell.bandwidth;
	}

	std::cout << "Total bandwidth: " << bandwidth << " B/s, " << bandwidth / 1e9 << " GB/s\n";

	for (auto& thread : threads)
		thread.join();
}
