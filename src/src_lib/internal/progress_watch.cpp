#include <utility>
#include <unordered_map>
#include <iostream>
#include "progress_watch.hpp"
#include "http.hpp"

namespace yolo
{
	void log(const std::string_view& message);
}

namespace yolo::internal
{

	static std::string_view check_server_str(const std::string_view& server)
	{
		if(server.empty())
		{
			return "";
		}
		if(!server.starts_with("http"))
		{
			// not an url.
			return "";
		}
		return server;
	}


	progress_watch::progress_watch(const std::string_view& server, std::filesystem::path weights_folder_path)
		: m_server(check_server_str(server))
		, m_weights_folder_path(std::move(weights_folder_path))
	{
		if(!ok())
		{
			return;
		}

		m_p_thread = std::make_unique<std::thread>([this](){thread_main();});
	}

	progress_watch::~progress_watch()
	{
		m_should_exit = true;
		if(m_p_thread != nullptr)
		{
			m_p_thread->join();
		}
	}

	std::unique_ptr<progress_watch> progress_watch::create(const std::string_view& server, const std::filesystem::path& weights_folder_path)
	{
		auto ptr = std::make_unique<progress_watch>(server, weights_folder_path);
		if(!ptr->ok())
		{
			ptr = nullptr;
		}
		return ptr;
	}

	static void notify(const std::string_view& server, const std::filesystem::path& filepath)
	{
		http::upload(std::string(server) + "/upload", filepath, "data");
	}

	void progress_watch::check_file(const std::filesystem::path& filepath, bool remove_once_send)
	{
		const std::string filepath_str = filepath.string();
		auto match = m_detected_files.find(filepath_str);
		if(match == m_detected_files.end())
		{
			log("watch - new file detected: '" + filepath_str + "'");
			internal::watch_file_data data = {
					.path = filepath,
					.notified = false,
					.time_at_discovery = std::chrono::system_clock::now(),
					.last_write_time = std::nullopt
			};
			m_detected_files.insert({filepath_str, data});
		}
		else
		{
			internal::watch_file_data& data = match->second;

			// reset if file updated
			if(data.last_write_time != std::nullopt && *data.last_write_time != std::filesystem::last_write_time(filepath))
			{
				data = internal::watch_file_data {
						.path = data.path,
						.notified = false,
						.time_at_discovery = std::chrono::system_clock::now(),
						.last_write_time = std::nullopt
				};
			}

			// notify
			if(!data.notified && std::chrono::system_clock::now() - data.time_at_discovery > std::chrono::seconds(10))
			{
				data.last_write_time = std::filesystem::last_write_time(filepath);
				notify(m_server, data.path);
				data.notified = true;
				log("watch - file notified: '" + filepath_str + "'");
				if(remove_once_send)
				{
					std::filesystem::remove(filepath);
					m_detected_files.erase(filepath_str);
					log("watch - file deleted for it is no longer needed locally: '" + filepath_str + "'");
				}
			}
		}
	}

	void progress_watch::thread_main()
	{
		while(!m_should_exit)
		{
			if(std::filesystem::exists(m_weights_folder_path))
			{
				for(const auto& file : std::filesystem::directory_iterator(m_weights_folder_path))
				{
					if(!file.is_directory())
					{
						const bool should_be_deleted_after_upload = file.path().string().ends_with(".weights");
						check_file(file.path(), should_be_deleted_after_upload);
					}
				}
			}
			if(std::filesystem::exists("./chart.png"))
			{
				check_file("./chart.png");
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}

	bool progress_watch::ok() const
	{
		return !m_server.empty();
	}

}
