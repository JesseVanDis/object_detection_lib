#ifndef ALL_YOLO_SERVER_HPP
#define ALL_YOLO_SERVER_HPP

#include <memory>
#include <filesystem>
#include <thread>

namespace httplib
{
	class Server;
}

namespace yolo::server
{
	class server_internal_thread;

	struct init_args
	{
		std::filesystem::path images_and_txt_annotations_folder;
		std::filesystem::path weights_folder_path;
		int port = 8080;
	};

	class server_internal
	{
		public:
			explicit server_internal(init_args&& init_args);
			~server_internal();

			[[nodiscard]] bool is_running() const;

		private:
			const init_args m_init_args;
			std::unique_ptr<server_internal_thread> m_p_internal;
			std::unique_ptr<std::thread> m_p_thread;
			bool m_ok = false;
	};

	class server_internal_thread
	{
		public:
			explicit server_internal_thread(const init_args& init_args);
			~server_internal_thread();

			[[nodiscard]] bool is_running() const;
			void start();
			void close();

		private:
			const init_args& m_init_args;
			std::unique_ptr<httplib::Server> m_p_server;
	};
}

#endif //ALL_YOLO_SERVER_HPP
