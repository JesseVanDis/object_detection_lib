#ifdef MINIZIP_FOUND
#include <yolo.hpp>
#include <httplib.h>
#include <minizip/zip.h>
#include "server.hpp"
#include "internal.hpp"

namespace yolo
{
	void log(const std::string_view& message);
}

namespace yolo::server
{
	server::server(std::unique_ptr<server_internal>&& v) : m_internal(std::move(v)) {}
	server::~server() = default;

	static bool create_zip_file(const std::filesystem::path& dest_filename, const std::vector<std::filesystem::path>& files_to_zip);

	inline void read_file(const std::string &path, std::string &out) {
		std::ifstream fs(path, std::ios_base::binary);
		fs.seekg(0, std::ios_base::end);
		auto size = fs.tellg();
		fs.seekg(0);
		out.resize(static_cast<size_t>(size));
		fs.read(&out[0], static_cast<std::streamsize>(size));
	}

	server_internal_thread::server_internal_thread(const init_args& init_args)
			: m_init_args(init_args)
	{
		m_p_server = std::make_unique<httplib::Server>();

		m_p_server->Get("/test", [](const httplib::Request&, httplib::Response& res)
		{
			res.set_content("If you see this, the server is running", "text/plain");
		});

		m_p_server->Get("/get_images_list", [this](const httplib::Request&, httplib::Response& res)
		{
			std::stringstream ss;
			size_t image_index = 0;
			bool is_first = true;
			for(const auto& v : std::filesystem::directory_iterator(m_init_args.images_and_txt_annotations_folder))
			{
				if(v.path().extension() == ".txt")
				{
					std::string filename = v.path().filename().replace_extension("").string();
					if(!is_first)
					{
						ss << std::endl;
					}
					ss << image_index << ":" << filename;
					is_first = false;
					image_index++;
				}
			}
			res.set_content(ss.str(), "text/plain");
		});

		m_p_server->Get("/get_images", [this](const httplib::Request& t, httplib::Response& res)
		{
			if(!t.has_param("from"))
			{
				res.set_content("Error: missing 'from' param. this param must contain the starting index of the images range you'd like to obtain. (the first column of 'host:post/get_images_list')", "text/plain");
				return;
			}
			if(!t.has_param("to"))
			{
				res.set_content("Error: missing 'to' param. this param must contain the ending index of the images range you'd like to obtain. (the first column of 'host:post/get_images_list')", "text/plain");
				return;
			}
			std::string from_str = t.get_param_value("from");
			std::string to_str = t.get_param_value("to");
			unsigned int from = 0;
			unsigned int to = 0;
			try
			{
				from = std::stoul(from_str);
				to = std::stoul(to_str);
			}
			catch (...)
			{
				res.set_content("Error: failed to parse '" + from_str + "' or '" + to_str + "' to int", "text/plain");
				return;
			}

			std::vector<std::filesystem::path> files_to_send;
			unsigned int image_index = 0;
			for(const auto& v : std::filesystem::directory_iterator(m_init_args.images_and_txt_annotations_folder))
			{
				if(v.path().extension() == ".txt")
				{
					if(image_index < from)
					{
						image_index++;
						continue;
					}
					if(image_index > to)
					{
						break;
					}
					if(auto image_path = internal::find_related_image_filepath(v.path()))
					{
						files_to_send.push_back(v.path());
						files_to_send.push_back(*image_path);
					}
					else
					{
						log("Warning: Failed to find related image to '" + v.path().string() + "'");
					}
					image_index++;
				}
			}

			const auto zip_path = std::filesystem::temp_directory_path() / "data.zip";
			if(std::filesystem::exists(zip_path))
			{
				std::filesystem::remove(zip_path);
			}
			if(!create_zip_file(zip_path, files_to_send))
			{
				res.set_content("Error: failed to zip images of given range", "text/plain");
			}

			res.set_header("Content-Type", "application/zip");
			read_file(zip_path.string(), res.body);
			res.status = 200;
		});
	}

	void server_internal_thread::start()
	{
		m_p_server->listen("0.0.0.0", m_init_args.port);
	}

	server_internal_thread::~server_internal_thread()
	{
		close();
		log("Server closed");
	}

	bool server_internal_thread::is_running() const
	{
		return m_p_server->is_running();
	}

	void server_internal_thread::close()
	{
		m_p_server->stop();
	}

	server_internal::server_internal(init_args&& init_args)
		: m_init_args(std::move(init_args))
	{
		log("Starting server...");

		std::atomic_bool started = false;
		m_p_thread = std::make_unique<std::thread>([this, &started]()
				{
					m_p_internal = std::make_unique<server_internal_thread>(m_init_args);
					started = true;
					m_p_internal->start();
				});

		while(!started)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		const auto start_time = std::chrono::system_clock::now();
		bool ok = true;
		while(!m_p_internal->is_running())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
			if(std::chrono::system_clock::now() - start_time > std::chrono::seconds(10))
			{
				log("Starting server... Reached timeout of 10 secs. Probably wont start.");
				ok = false;
				break;
			}
		}
		m_ok = ok;
		log(m_ok ? ("Starting server... Ok! server is running on port '" + std::to_string(m_init_args.port) +  "'") : "Starting server... Failed.");
	}

	server_internal::~server_internal()
	{
		m_p_internal->close();
		m_p_thread->join();
		log("Server closed");
	}

	bool server_internal::is_running() const
	{
		return m_ok;
	}

	static bool create_zip_file(const std::filesystem::path& dest_filename, const std::vector<std::filesystem::path>& files_to_zip)
	{
		std::string dest_filename_str = dest_filename.string();
		const char* dest_filename_cstr = dest_filename_str.c_str();

		zipFile zf = zipOpen(dest_filename_cstr, APPEND_STATUS_CREATE);
		if (zf == nullptr)
		{
			return false;
		}

		bool _return = true;
		for (size_t i = 0; i < files_to_zip.size(); i++)
		{
			const std::string path = files_to_zip[i].string();
			std::fstream file(path.c_str(), std::ios::binary | std::ios::in);
			if (file.is_open())
			{
				file.seekg(0, std::ios::end);
				long size = file.tellg();
				file.seekg(0, std::ios::beg);

				std::vector<char> buffer(size);
				if (size == 0 || file.read(&buffer[0], size))
				{
					zip_fileinfo zfi = {};
					auto fileName = files_to_zip[i].filename().string(); //path.substr(path.rfind('\\')+1);

					if (ZIP_OK == zipOpenNewFileInZip(zf, std::string(fileName.begin(), fileName.end()).c_str(), &zfi, nullptr, 0, nullptr, 0, nullptr, Z_DEFLATED, Z_NO_COMPRESSION))
					{
						if (zipWriteInFileInZip(zf, size == 0 ? "" : &buffer[0], size))
							_return = false;

						if (zipCloseFileInZip(zf))
							_return = false;

						file.close();
						continue;
					}
				}
				file.close();
			}
			_return = false;
		}

		if (zipClose(zf, nullptr))
		{
			return false;
		}

		if (!_return)
		{
			return false;
		}
		return true;
	}

}

#endif