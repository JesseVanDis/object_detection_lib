#define STBI_NO_TGA

#include <iostream>
#include <fstream>
#include <darknet.h>
#include <thread>
#include <map>
#include "internal.hpp"
#include "http.hpp"
#include "zip.hpp"

#ifdef GPU_SHOW_INFO
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#endif


namespace yolo
{
	void(*s_log_function)(const std::string_view& message) = nullptr;

	void log(const std::string_view& message)
	{
		if(s_log_function != nullptr)
		{
			s_log_function(message);
		}
		else
		{
			std::cout << "obj_det: " << message << std::endl;
		}
	}


	namespace internal
	{
		void set_log_callback(void(*log_function)(const std::string_view& message))
		{
			s_log_function = log_function;
		}

		bool write_yolo_data(const std::filesystem::path& dest_filepath, const yolo_data& data)
		{
			std::filesystem::path folder = dest_filepath;
			folder.remove_filename();
			if(!std::filesystem::exists(folder))
			{
				std::filesystem::create_directories(folder);
			}

			std::ofstream file;
			file.open (dest_filepath);
			if(!file.is_open())
			{
				return false;
			}

			file     <<         "classes = " << data.classes;
			file     << "\n" << "train = " << std::filesystem::weakly_canonical(std::filesystem::absolute(data.train)).string();
			file     << "\n" << "valid = " << std::filesystem::weakly_canonical(std::filesystem::absolute(data.valid)).string();
			file     << "\n" << "names = " << std::filesystem::weakly_canonical(std::filesystem::absolute(data.names)).string();
			if(data.backup.has_value())
			{
				file << "\n" << "backup = " << std::filesystem::weakly_canonical(std::filesystem::absolute(*data.backup)).string();
			}
			file.close();
			return true;
		}

		std::optional<std::filesystem::path> find_latest_backup_weights(const std::filesystem::path& folder_path)
		{
			std::optional<std::filesystem::path> latest;
			std::optional<std::filesystem::file_time_type> latest_write_time;
			if(!std::filesystem::exists(folder_path))
			{
				return std::nullopt;
			}
			for(const auto& filepath : std::filesystem::directory_iterator(folder_path))
			{
				if(filepath.path().extension() == ".weights")
				{
					if(!latest_write_time.has_value())
					{
						latest = filepath.path();
						latest_write_time = filepath.last_write_time();
					}
					else if(*latest_write_time > filepath.last_write_time())
					{
						latest = filepath.path();
						latest_write_time = filepath.last_write_time();
					}
				}
			}
			return latest;
		}

		std::optional<std::filesystem::path> obtain_starting_weights(const std::string& pretrained_weights_url, const std::optional<std::filesystem::path>& backup_path, const std::optional<std::filesystem::path>& download_target_path)
		{
			const std::filesystem::path tmp_path = std::filesystem::temp_directory_path();

			auto last_update = std::chrono::system_clock::now();
			auto download_update = [&](const yolo::http::progress& progress)
			{
				const auto now = std::chrono::system_clock::now();
				const auto time_since_last_update = now - last_update;
				if(time_since_last_update > std::chrono::seconds(2))
				{
					last_update = now;
					log("downloaded: '" + std::to_string(progress.bytes_written / 1'000'000) + "mb'. progress: " + std::to_string(progress.progress_perc() * 100.0) + "%...");
				}
			};

			const auto pretrained_model_path = download_target_path.value_or(tmp_path / "weights" / "pretrained" / "darknet53.conv.74");
			const auto latest_backup_weights = backup_path.has_value() ? find_latest_backup_weights(*backup_path) : std::nullopt;
			if(latest_backup_weights.has_value())
			{
				log("selecting '" + latest_backup_weights->string() + "' for starting weights");
				return *latest_backup_weights;
			}
			else
			{
				if(!std::filesystem::exists(pretrained_model_path))
				{
					log("Initiating download from '" + pretrained_weights_url + "'...");
					if(!yolo::http::download(pretrained_weights_url, pretrained_model_path, true, download_update))
					{
						log("Failed to download pretrained model");
						return std::nullopt;
					}
					log("Download done!");
				}
				log("selecting '" + pretrained_model_path.string() + "' for starting weights");
			}
			return pretrained_model_path;
		}

		static std::string remove_extension(const std::string& filename)
		{
			size_t lastdot = filename.find_last_of('.');
			if (lastdot == std::string::npos) return filename;
			return filename.substr(0, lastdot);
		}

		std::optional<std::filesystem::path> find_related_image_filepath(const std::filesystem::path& filepath_txt)
		{
			const std::string path_without_extension = remove_extension(filepath_txt);
			if(std::filesystem::exists(path_without_extension + ".png"))
			{
				return path_without_extension + ".png";
			}
			if(std::filesystem::exists(path_without_extension + ".jpg"))
			{
				return path_without_extension + ".jpg";
			}
			return std::nullopt;
		}

		//static std::array<char, 128> str_to_c(const std::string_view& str) { std::array<char, 128> v = {0}; strncpy(v.data(), str.data(), v.size()); return v; }
		static std::array<char, 128> str_to_c(const std::filesystem::path& str)
		{
			std::array<char, 128> v = {0};
			const auto path_str = str.string();
			if(path_str.size() < v.size())
			{
				std::copy(path_str.begin(), path_str.end(), v.begin());
				v[path_str.size()] = '\0';
			}
			else
			{
				log("Failed to convert '" + path_str + "' to c_str");
			}
			return v;
		}

		struct init_darknet_result
		{
			int gpu_index = -1;
			int *gpus = nullptr;
			int ngpus = 0;
		};

		static init_darknet_result init_darknet();

		bool start_darknet_training(const std::filesystem::path& model_cfg_data, const cfg::cfg& model_cfg, const std::filesystem::path& starting_weights, const darknet_training_args& args)
		{
			const std::filesystem::path tmp_path = std::filesystem::temp_directory_path();
			const auto model_cfg_filepath = tmp_path / "model.cfg";
			model_cfg.save(model_cfg_filepath);

			auto darknet = init_darknet();

			// !./darknet detector train data/yolo.data cfg/yolov3_custom_train.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map

			auto model_cfg_data_c = str_to_c(model_cfg_data);
			auto model_cfg_c = str_to_c(model_cfg_filepath);
			auto starting_weights_c = str_to_c(starting_weights);
			auto chart_path_c = str_to_c(args.chart_path.has_value() ? *args.chart_path : "");

			train_detector(
					model_cfg_data_c.data(),
					model_cfg_c.data(),
					starting_weights_c.data(),
					darknet.gpus,
					darknet.ngpus,
					args.should_clear ? 1 : 0,
					args.dont_show ? 1 : 0,
					args.map ? 1 : 0,
					args.thresh,
					args.iou_thresh,
					args.mjpeg_port,
					args.show_imgs ? 1 : 0,
					args.benchmark_layers ? 1 : 0,
					chart_path_c[0] == '\0' ? nullptr : chart_path_c.data());

			return true;
		}

		static bool parse_images_list_line(std::string& line, unsigned int& index_out, std::string& image_name_out)
		{
			line.push_back('\0');
			char* line_cstr = line.data();
			for(size_t i=0; i<line.size(); i++)
			{
				if(line_cstr[i] == ':')
				{
					line_cstr[i] = '\0';
					index_out = atoll(line_cstr);
					image_name_out = &line_cstr[i+1];
					return true;
				}
			}
			return false;
		}

		static std::optional<std::map<unsigned int, std::string>> download_images_list(const std::string_view& server_and_port)
		{
			std::vector<uint8_t> images_list_data;
			if(!http::download(std::string(server_and_port) + "/get_images_list", images_list_data))
			{
				return std::nullopt;
			}
			std::map<unsigned int, std::string> map;

			std::string line;
			unsigned int index = ~0u;
			std::string image_name;
			for(auto& v : images_list_data)
			{
				if(v != '\n')
				{
					line.push_back((char)v);
				}
				else
				{
					if(parse_images_list_line(line, index, image_name))
					{
						map.insert({index, image_name});
					}
					line.clear();
				}
			}
			if(!line.empty())
			{
				if(parse_images_list_line(line, index, image_name))
				{
					map.insert({index, image_name});
				}
			}
			return map;
		}

		bool obtain_trainingdata_server(const std::string_view& server_and_port, const std::filesystem::path& dest_path, const std::function<void(const obtain_data_from_server_progress& progress)>& progress_callback)
		{
			static const size_t s_max_images_per_batch = 100;
			if(!std::filesystem::exists(dest_path))
			{
				std::filesystem::create_directories(dest_path);
			}

			auto images_list = download_images_list(server_and_port);
			if(!images_list.has_value() || images_list->empty())
			{
				return false;
			}

			// obtain missing images
			std::vector<unsigned int> missing_images;
			{
				for(const auto& v : *images_list)
				{
					if(!std::filesystem::exists(dest_path / (v.second + ".txt")))
					{
						missing_images.push_back(v.first);
					}
				}
				if(missing_images.empty())
				{
					return true; // all images already obtained
				}
				std::sort(missing_images.begin(), missing_images.end());
			}

			// split 'missing_images' to batches
			std::vector<std::pair<unsigned int, unsigned int>> batches;
			{
				unsigned int from = missing_images[0];
				unsigned int previous = ~0u;
				for(size_t i=0; i<missing_images.size(); i++)
				{
					bool is_end_of_batch = false;
					is_end_of_batch |= (missing_images[i] != (previous+1));
					is_end_of_batch |= (missing_images[i] - from) >= (s_max_images_per_batch-1);
					is_end_of_batch |= i == (missing_images.size()-1);
					previous = missing_images[i];

					if(is_end_of_batch)
					{
						batches.emplace_back(from, missing_images[i]);
						from = missing_images[std::min((int)i+1, (int)missing_images.size()-1)];
					}
				}
			}

			// download the missing image batches
			{
				size_t total_num_images = 0;
				size_t num_images_done = 0;
				for(const auto& v : batches)
				{
					total_num_images += (v.second - v.first)+1;
				}
				for(const auto& v : batches)
				{
					std::string filename = std::to_string(v.first) + "_" + std::to_string(v.second) + ".zip";
					const std::filesystem::path dest_file = dest_path / filename;
					http::download(std::string(server_and_port) + "/get_images?from=" + std::to_string(v.first) + "&to=" + std::to_string(v.second), dest_file, true, std::nullopt, true);
					if(std::filesystem::exists(dest_file))
					{
						zip::extract_zip_file(dest_file, dest_path);
						std::filesystem::remove(dest_file);
					}
					num_images_done += (v.second - v.first)+1;

					obtain_data_from_server_progress progress = {
							.num_images_obtained = num_images_done,
							.total_num_images = total_num_images
					};
					progress_callback(progress);
				}
			}

			return true;
		}

		bool obtain_trainingdata_server(const std::string_view& server_and_port, const std::filesystem::path& dest_path)
		{
			auto last_status_update = std::chrono::system_clock::now();
			log("downloading data from '" + std::string(server_and_port) + "'...");
			const bool ok = obtain_trainingdata_server(server_and_port, dest_path, [&](const obtain_data_from_server_progress& progress)
			{
				const auto now = std::chrono::system_clock::now();
				if(now - last_status_update > std::chrono::seconds(1))
				{
					log("downloading data from '" + std::string(server_and_port) + "'... [" + std::to_string(progress.num_images_obtained) + "\\" + std::to_string(progress.total_num_images) + "]");
				}
			});
			if(!ok)
			{
				log("downloading data from '" + std::string(server_and_port) + "'... Failed!");
				return false;
			}
			log("downloading data from '" + std::string(server_and_port) + "'... Done!");
			return true;
		}

#ifdef GPU_SHOW_INFO // Hack: not part of darknet.h, it resides somewhere in 'dark_cuda.c' but still useful function. hopefully it keeps existing. getting linker errors on colab though...
		static void show_cuda_cudnn_info()
		{
			int cuda_version = 0, cuda_driver_version = 0, device_count = 0;
			cudaRuntimeGetVersion(&cuda_version);
			cudaDriverGetVersion(&cuda_driver_version);
			log("  CUDA-version: " + std::to_string(cuda_version) + " (" + std::to_string(cuda_driver_version) + ")");
			if(cuda_version > cuda_driver_version)
			{
				log("Warning: CUDA-version is higher than Driver-version!");
			}
		#ifdef CUDNN
			log("  cuDNN: " + std::to_string(CUDNN_MAJOR) + "." + std::to_string(CUDNN_MINOR) + "." + std::to_string(CUDNN_PATCHLEVEL));
		#endif  // CUDNN
		#ifdef CUDNN_HALF
			log("  CUDNN_HALF=1");
		#endif  // CUDNN_HALF
			cudaGetDeviceCount(&device_count);
			log("  GPU count: " + std::to_string(device_count));
		}
#endif

		static init_darknet_result init_darknet()
		{
			static init_darknet_result s_result;
			static bool s_initialized = false;
			if(s_initialized)
			{
				return s_result;
			}

#ifndef GPU
			gpu_index = -1;
			log(" GPU isn't used");
			init_cpu();
#else
#ifdef GPU_SHOW_INFO
			show_cuda_cudnn_info();
#endif
#endif
			s_initialized = true;

			s_result.gpu_index = gpu_index;
			s_result.gpus = &s_result.gpu_index;
			s_result.ngpus = 1;

			return s_result;
		}
	}
}
