#include <fstream>
#include <string>
#include <iostream>
#include <curl/curl.h>
#include <cassert>
#include "http.hpp"

namespace yolo
{
	void log(const std::string_view& message);

	namespace http
	{
		struct progress_callback_args
		{
			double dltotal = 0;
			double dlnow = 0;
			double ultotal = 0;
			double ulnow = 0;
		};

		struct writer_dest
		{
			std::vector<uint8_t>* p_data;
			std::ofstream* p_ofstream;
			const std::function<void(const progress& progress)>* p_callback;
			progress_callback_args* p_progress_args;

			size_t num_bytes_written = 0;
		};

		static int writer(char *data, size_t size, size_t nmemb, writer_dest* p_writer_dest)
		{
			int result = 0;
			if (p_writer_dest != nullptr)
			{
				p_writer_dest->num_bytes_written += size * nmemb;

				if(p_writer_dest->p_data != nullptr)
				{
					p_writer_dest->p_data->insert(p_writer_dest->p_data->end(), &data[0], &data[size * nmemb]);
				}
				if(p_writer_dest->p_ofstream != nullptr)
				{
					p_writer_dest->p_ofstream->write(data, size * nmemb);
				}
				if(p_writer_dest->p_callback != nullptr)
				{
					assert(p_writer_dest->p_progress_args != nullptr);
					if(p_writer_dest->p_progress_args == nullptr)
					{
						// should not get here actually
						const progress progress =
								{
										.bytes_written = p_writer_dest->num_bytes_written,
										.progress_total = 0,
										.progress_now = 0
								};
						(*p_writer_dest->p_callback)(progress);
					}
					else
					{
						const progress progress =
								{
										.bytes_written = p_writer_dest->num_bytes_written,
										.progress_total = p_writer_dest->p_progress_args->dltotal,
										.progress_now = p_writer_dest->p_progress_args->dlnow
								};
						(*p_writer_dest->p_callback)(progress);
					}
				}
				result = size * nmemb;
			}
			return result;
		}


		static size_t curl_progress_callback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow)
		{
			progress_callback_args* user_data = (progress_callback_args*)clientp;
			user_data->dltotal = dltotal;
			user_data->dlnow = dlnow;
			user_data->ultotal = ultotal;
			user_data->ulnow = ulnow;
			return 0;
		}


		bool download(const std::string_view& url, std::optional<std::filesystem::path> dest_filepath, std::vector<uint8_t>* dest_data, bool overwrite, const std::optional<std::function<void(const progress& progress)>>& progress_callback, bool silent)
		{
			char errorBuffer[CURL_ERROR_SIZE];

			if(!silent)
			{
				log("Downloading '" + std::string(url) + "'...");
			}

			CURL *curl;
			CURLcode result;
			curl = curl_easy_init();

			if(curl == nullptr)
			{
				log("Failed to init CURL");
				return false;
			}

			struct dest_url
			{
				std::filesystem::path dest_filepath;
				std::string temp_filepath;
				std::ofstream file;
			};

			std::optional<dest_url> dest_url_opt;

			if(dest_filepath.has_value())
			{
				dest_url_opt = dest_url {
						.dest_filepath = *dest_filepath,
						.temp_filepath = dest_filepath->string() + ".tmp",
						.file = {}
				};
				if(std::filesystem::exists(dest_url_opt->temp_filepath))
				{
					std::filesystem::remove(dest_url_opt->dest_filepath);
				}
				if(std::filesystem::exists(dest_url_opt->dest_filepath))
				{
					if(!overwrite)
					{
						log("Download failed. '" + dest_url_opt->dest_filepath.string() + "' already exists");
						return false;
					}
					else
					{
						std::filesystem::remove(dest_url_opt->dest_filepath);
					}
				}
				const auto folder = dest_url_opt->dest_filepath.parent_path();
				if(!std::filesystem::exists(folder))
				{
					std::filesystem::create_directories(folder);
				}
				std::ofstream file(dest_url_opt->temp_filepath, std::ios::out | std::ios::binary);
				if(!file)
				{
					log("Download failed. Failed to open '" + dest_url_opt->temp_filepath + "'");
					return false;
				}
				dest_url_opt->file = std::move(file);
			}

			// curl stuff
			{
				progress_callback_args progress_args;

				writer_dest dest_args = {
						.p_data = dest_data,
						.p_ofstream = dest_url_opt.has_value() ? &dest_url_opt->file : nullptr,
						.p_callback = progress_callback.has_value() ? &(*progress_callback) : nullptr,
						.p_progress_args = &progress_args
				};

				std::string url_data(url);

				// Now set up all of the curl options
				curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer);
				curl_easy_setopt(curl, CURLOPT_URL, url_data.c_str());
				curl_easy_setopt(curl, CURLOPT_HEADER, 0);
				curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
				curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0);
				curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &progress_args);
				curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, curl_progress_callback);
				curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writer);
				curl_easy_setopt(curl, CURLOPT_WRITEDATA, &dest_args);


				// Attempt to retrieve the remote page
				result = curl_easy_perform(curl);

				// Always cleanup
				curl_easy_cleanup(curl);

			}

			if(dest_url_opt.has_value())
			{
				dest_url_opt->file.close();

				// Did we succeed?
				if (result == CURLE_OK)
				{
					if(std::filesystem::exists(dest_url_opt->temp_filepath))
					{
						std::filesystem::rename(dest_url_opt->temp_filepath, dest_url_opt->dest_filepath);
					}
				}
				else
				{
					if(std::filesystem::exists(dest_url_opt->dest_filepath))
					{
						std::filesystem::remove(dest_url_opt->dest_filepath);
					}
					if(std::filesystem::exists(dest_url_opt->temp_filepath))
					{
						std::filesystem::remove(dest_url_opt->temp_filepath);
					}
					log("Download failed. CURL error: '" + std::string(errorBuffer) + "'");
					return false;
				}
			}

			if(!silent)
			{
				log("Downloading '" + std::string(url) + "'... Done!");
			}
			return true;
		}

		bool download(const std::string_view& url, std::vector<uint8_t>& dest, const std::optional<std::function<void(const progress& progress)>>& progress_callback, bool silent)
		{
			return download(url, std::nullopt, &dest, false, progress_callback, silent);
		}

		bool download(const std::string_view& url, const std::filesystem::path& dest_filepath, bool overwrite, const std::optional<std::function<void(const progress& progress)>>& progress_callback, bool silent)
		{
			return download(url, dest_filepath, nullptr, overwrite, progress_callback, silent);
		}

		std::optional<std::string> download_str(const std::string_view& url, bool silent)
		{
			std::vector<uint8_t> result;
			download(url, result, std::nullopt, silent);
			result.push_back('\0');
			std::string result_str = (const char*)result.data();
			return result_str.empty() ? std::nullopt : std::make_optional(result_str);
		}

		std::optional<std::string> fetch_public_ipv4()
		{
			return download_str("https://api.ipify.org");
		}

	}
}
