#include <fstream>
#include <string>
#include <iostream>
#include <curl/curl.h>
#include <cassert>
#include <sys/stat.h>
#include <iterator>
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

		bool download(const std::string_view& url, std::optional<std::filesystem::path> dest_filepath, std::vector<uint8_t>* dest_data, const additional_download_args& args)
		{
			char errorBuffer[CURL_ERROR_SIZE];

			if(!args.silent)
			{
				log("Downloading '" + std::string(url) + "'...");
			}

			CURL *curl;
			CURLcode result;
			curl = curl_easy_init();

			if(curl == nullptr)
			{
				if(!args.on_error("Failed to init CURL", 0))
				{
					log(errorBuffer);
				}
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
					if(!args.overwrite)
					{
						snprintf(errorBuffer, sizeof(errorBuffer)-1, "Download failed. '%s' already exists", dest_url_opt->dest_filepath.c_str());
						if(!args.on_error(errorBuffer, 0))
						{
							log(errorBuffer);
						}
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
					snprintf(errorBuffer, sizeof(errorBuffer)-1, "Download failed. Failed to open '%s'", dest_url_opt->temp_filepath.c_str());
					if(!args.on_error(errorBuffer, 0))
					{
						log(errorBuffer);
					}
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
						.p_callback = args.progress_callback.has_value() ? &(*args.progress_callback) : nullptr,
						.p_progress_args = &progress_args
				};

				std::string url_data(url);

				// Now set up all the curl options
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

				long http_code = 0;
				curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &http_code);

				// Did we succeed?
				if (result == CURLE_OK && http_code == 200)
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

					char error_msg[sizeof(errorBuffer) + 256];
					snprintf(error_msg, sizeof(error_msg)-1, "Download failed. CURL error: '%s' http code: %d", errorBuffer, (int)http_code);
					if(!args.on_error(error_msg, http_code))
					{
						log(error_msg);
					}
					return false;
				}
			}

			if(!args.silent)
			{
				log("Downloading '" + std::string(url) + "'... Done!");
			}
			return true;
		}

		bool download(const std::string_view& url, const std::filesystem::path& dest_filepath, const additional_download_args& args)
		{
			return download(url, dest_filepath, nullptr, args);
		}

		bool download(const std::string_view& url, std::vector<uint8_t>& dest, const additional_download_args& args)
		{
			return download(url, std::nullopt, &dest, args);
		}

		static size_t upload_write_callback(void * buffer, size_t size, size_t count, void * user)
		{
			size_t numBytes = size * count;
			static_cast<std::string*>(user)->append(static_cast<char*>(buffer), 0, numBytes);
			return numBytes;
		}

		bool upload(const std::string_view& url, const std::filesystem::path& file_to_upload, const std::string_view& name, bool silent)
		{
			const std::string file_to_upload_str = file_to_upload.string();
			const std::string name_str(name);

			char errorBuffer[CURL_ERROR_SIZE];

			if(!silent)
			{
				log("Uploading '" + std::string(file_to_upload_str) + "' to '" + std::string(url) + "'...");
			}

			if(std::filesystem::is_directory(file_to_upload))
			{
				log("Upload failed. '" + file_to_upload_str + "' is a directory, not a file");
				return false;
			}
			if(!std::filesystem::exists(file_to_upload))
			{
				log("Upload failed. '" + file_to_upload_str + "' does not exist");
				return false;
			}

			CURL *curl;
			CURLcode result;
			curl = curl_easy_init();

			if(curl == nullptr)
			{
				log("Failed to init CURL");
				return false;
			}

			// curl stuff
			{
				std::string url_data(url);

				//curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0);
				//curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

				curl_httppost *formpost = nullptr;
				curl_httppost *lastptr = nullptr;

				curl_formadd(&formpost,
							 &lastptr,
							 CURLFORM_COPYNAME, name_str.c_str(),
							 CURLFORM_FILE, file_to_upload_str.c_str(),
							 CURLFORM_CONTENTTYPE, "multipart/form-data",
							 CURLFORM_END);

				curl_slist *headerlist = curl_slist_append(nullptr, "Expect:");

				curl_easy_setopt(curl, CURLOPT_URL, url_data.c_str());

				curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
				curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);

				std::string reponse;
				curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, upload_write_callback);
				curl_easy_setopt(curl, CURLOPT_WRITEDATA, &reponse);
				if(!silent)
				{
					log("Received response: " + reponse);
				}

				result = curl_easy_perform(curl);

				curl_formfree(formpost);
				curl_slist_free_all(headerlist);
			}

			// Did we succeed?
			if (result != CURLE_OK)
			{
				log("Download failed. CURL error: '" + std::string(errorBuffer) + "'");
				return false;
			}

			if(!silent)
			{
				log("Uploading '" + std::string(file_to_upload_str) + "' to '" + std::string(url) + "'... Done!");
			}
			return true;
		}

		std::optional<std::string> download_str(const std::string_view& url, bool silent)
		{
			std::vector<uint8_t> result;
			download(url, result, http::additional_download_args{.silent = silent});
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
