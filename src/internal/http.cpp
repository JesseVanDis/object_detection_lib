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
					for(size_t i=0; i<size * nmemb; i++)
					{
						p_writer_dest->p_data->push_back(data[i]);
					}
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

		bool download(const std::string_view& url, const std::filesystem::path& dest_filepath, bool overwrite, const std::optional<std::function<void(const progress& progress)>>& progress_callback)
		{
			char errorBuffer[CURL_ERROR_SIZE];

			auto temp_filepath = dest_filepath.string() + ".tmp";
			log("Downloading '" + std::string(url) + "'...");

			if(std::filesystem::exists(temp_filepath))
			{
				std::filesystem::remove(dest_filepath);
			}
			if(std::filesystem::exists(dest_filepath))
			{
				if(!overwrite)
				{
					log("Download failed. '" + dest_filepath.string() + "' already exists");
					return false;
				}
				else
				{
					std::filesystem::remove(dest_filepath);
				}
			}

			CURL *curl;
			CURLcode result;
			curl = curl_easy_init();
			if (curl)
			{
				const auto folder = dest_filepath.parent_path();
				if(!std::filesystem::exists(folder))
				{
					std::filesystem::create_directories(folder);
				}
				std::ofstream file(temp_filepath, std::ios::out | std::ios::binary);
				if(!file)
				{
					log("Download failed. Failed to open '" + temp_filepath + "'");
					return false;
				}

				progress_callback_args progress_args;

				writer_dest dest_args = {
						.p_data = nullptr,
						.p_ofstream = &file,
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

				file.close();

				// Did we succeed?
				if (result == CURLE_OK)
				{
					if(std::filesystem::exists(temp_filepath))
					{
						std::filesystem::rename(temp_filepath, dest_filepath);
					}
					return true;
				}
				else
				{
					if(std::filesystem::exists(dest_filepath))
					{
						std::filesystem::remove(dest_filepath);
					}
					if(std::filesystem::exists(temp_filepath))
					{
						std::filesystem::remove(temp_filepath);
					}
					log("Download failed. CURL error: '" + std::string(errorBuffer) + "'");
					return false;
				}
			}
			else
			{
				log("Failed to init CURL");
				return false;
			}

			return true;
		}
	}
}
