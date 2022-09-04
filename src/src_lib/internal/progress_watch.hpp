#ifndef ALL_PROGRESS_WATCH_HPP
#define ALL_PROGRESS_WATCH_HPP

#include <string_view>
#include <filesystem>
#include <atomic>
#include <thread>

namespace yolo::internal
{
	namespace internal
	{
		struct watch_file_data
		{
			std::filesystem::path 	path;
			bool 					notified = false;
			std::chrono::system_clock::time_point time_at_discovery;
		};
	}

	class progress_watch
	{
		public:
			explicit progress_watch(const std::string_view& server, std::filesystem::path  weights_folder_path = "./weights");
			~progress_watch();

			static std::unique_ptr<progress_watch> create(const std::string_view& server, const std::filesystem::path& weights_folder_path = "./weights");

			[[nodiscard]] bool ok() const;

		private:
			void check_file(const std::filesystem::path& filepath);
			void thread_main();

			const std::string 				m_server;
			const std::filesystem::path 	m_weights_folder_path;
			std::atomic_bool 				m_should_exit = false;
			std::unique_ptr<std::thread>	m_p_thread;
			std::unordered_map<std::string, internal::watch_file_data> m_detected_files;
	};
}

#endif //ALL_PROGRESS_WATCH_HPP
