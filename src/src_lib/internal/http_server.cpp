#ifdef MINIZIP_FOUND
#define CPPHTTPLIB_THREAD_POOL_COUNT 2
#include <yolo.hpp>
#include <httplib.h>
#include "http_server.hpp"
#include "internal.hpp"
#include "zip.hpp"

namespace yolo
{
	void log(const std::string_view& message);
}

namespace yolo::http::server
{
	server::server(std::unique_ptr<server_internal>&& v) : m_internal(std::move(v)) {}
	server::~server() = default;

	inline void read_file(const std::string &path, std::string &out)
	{
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

		m_p_server->Get("/get_data_source", [this](const httplib::Request&, httplib::Response& res)
		{
			std::stringstream ss;
			size_t image_index = 0;
			bool is_first = true;

			if(m_init_args.data_source.starts_with("open_images,"))
			{
				ss << "open_images" << std::endl;
				ss << m_init_args.data_source << std::endl;
			}
			else
			{
				ss << "images_list" << std::endl;
				for(const auto& v : std::filesystem::directory_iterator(m_init_args.data_source))
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
			}
			res.set_content(ss.str(), "text/plain");
		});

		m_p_server->Get("/get_images", [this](const httplib::Request& t, httplib::Response& res)
		{
			if(!t.has_param("from"))
			{
				res.set_content("Error: missing 'from' param. this param must contain the starting index of the images range you'd like to obtain. (the first column of 'host:post/get_data_source')", "text/plain");
				return;
			}
			if(!t.has_param("to"))
			{
				res.set_content("Error: missing 'to' param. this param must contain the ending index of the images range you'd like to obtain. (the first column of 'host:post/get_data_source')", "text/plain");
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
			for(const auto& v : std::filesystem::directory_iterator(m_init_args.data_source))
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
			if(!zip::create_zip_file(zip_path, files_to_send))
			{
				res.set_content("Error: failed to zip images of given range", "text/plain");
			}

			res.set_header("Content-Type", "application/zip");
			read_file(zip_path.string(), res.body);
			res.status = 200;
		});

		m_p_server->Get("/latest_weights", [this](const httplib::Request&, httplib::Response& res)
		{
			std::optional<std::string> attached_filename;
			bool ok = false;
			if(auto path = m_init_args.latest_weights_filepath)
			{
				if(std::filesystem::exists(*path))
				{
					const auto zip_path = std::filesystem::temp_directory_path() / "data_weights.zip";
					if(std::filesystem::exists(zip_path))
					{
						std::filesystem::remove(zip_path);
					}
					if(!zip::create_zip_file(zip_path, {*path}))
					{
						res.set_content("Error: failed to zip images of given range", "text/plain");
					}
					res.set_header("Content-Type", "application/zip");
					read_file(zip_path.string(), res.body);
					res.status = 200;
					ok = true;
				}
			}
			if(!ok)
			{
				res.status = 204;
			}
		});

		m_p_server->Post("/upload", [&](const httplib::Request &req, httplib::Response &res, const httplib::ContentReader &content_reader)
		{
			bool ok = false;
			if (req.is_multipart_form_data())
			{
				std::string name;
				std::string filename;
				std::string content_type;
				std::vector<uint8_t> file_data;

				content_reader(
						[&](const httplib::MultipartFormData &file)
						{
							if(!file_data.empty())
							{
								handle_file_upload(name, filename, content_type, file_data);
							}
							name = file.name;
							filename = file.filename;
							content_type = file.content_type;
							file_data.clear();
							return true;
						},
						[&](const char *data, size_t data_length)
						{
							file_data.insert(file_data.end(), (const uint8_t*)data, (const uint8_t*)(data+data_length));
							return true;
						});

				if(!file_data.empty())
				{
					handle_file_upload(name, filename, content_type, file_data);
					ok = true;
				}
			}
			if(!ok)
			{
				log("Received post request, but not handled.");
			}
			res.set_content("Content received", "text/plain");
		});
	}

	void server_internal_thread::handle_file_upload(const std::string& name, const std::string& filename, const std::string& content_type, const std::vector<uint8_t>& content)
	{
		log("/upload invoked with name: '" + name + "', filename: '" + filename + "', content_type: '" + content_type + "' num bytes: " + std::to_string(content.size()));

		std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / filename;
		if(std::filesystem::exists(temp_dir))
		{
			std::filesystem::remove(temp_dir);
		}

		if(!std::filesystem::exists(m_init_args.weights_folder_path))
		{
			std::filesystem::create_directories(m_init_args.weights_folder_path);
		}

		// safely write it to a temporary folder. if interrupted it's ok for it to be corrupted.
		// when done, 'mv' it to the destination. ( should be safe )
		std::string temp_dir_str = temp_dir.string();
		{
			std::ofstream file(temp_dir_str.c_str(), std::ios::out | std::ios::binary);
			if(!file)
			{
				std::cerr << "Cannot open file!" << std::endl;
				return;
			}
			file.write((const char*)content.data(), content.size());
			file.close();
		}

		if(filename.ends_with(".weights"))
		{
			std::filesystem::path weights_path = m_init_args.weights_folder_path / filename;
			if(std::filesystem::exists(weights_path))
			{
				std::filesystem::remove(weights_path);
			}
			std::filesystem::rename(temp_dir, weights_path);
			log("'" + weights_path.string() + "' written.");

			// weights can be:
			//  model_10.weights
			//  model_20.weights
			//  model_100.weights
			//  model_last.weights
		}
		else if(filename.ends_with(".png"))
		{
			std::filesystem::path png_path = m_init_args.chart_png_path;
			if(std::filesystem::exists(png_path))
			{
				std::filesystem::remove(png_path);
			}
			std::filesystem::rename(temp_dir, png_path);
			log("'" + png_path.string() + "' written.");
			// chart can be: chart.png
		}
		else
		{
			log("don't know what to do with '" + filename + "'.");
		}
	}

	void server_internal_thread::start()
	{
		m_p_server->listen("0.0.0.0", m_init_args.port);
	}

	server_internal_thread::~server_internal_thread()
	{
		close();
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

}

#endif