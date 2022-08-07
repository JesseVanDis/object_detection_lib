#define STBI_NO_TGA

#include <iostream>
#include <fstream>
#include <darknet.h>
#include <thread>
#include "internal.hpp"
#include "http.hpp"

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
#else   // GPU
			if(gpu_index >= 0){
				cuda_set_device(gpu_index);
				CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
			}
			show_cuda_cudnn_info();
			cuda_debug_sync = find_arg(argc, argv, "-cuda_debug_sync");
		#ifdef CUDNN_HALF
			log(" CUDNN_HALF=1");
		#endif  // CUDNN_HALF
#endif  // GPU
			s_initialized = true;

			s_result.gpu_index = gpu_index;
			s_result.gpus = &s_result.gpu_index;
			s_result.ngpus = 1;

			return s_result;
		}
	}
}
