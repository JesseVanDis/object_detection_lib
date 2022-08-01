#define STBI_NO_TGA

#include <iostream>
#include <fstream>
#include <cstring>
#include <darknet.h>
#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>
#include <thread>
#include <atomic>
#include <execution>
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
			std::cout << "yolo: " << message << std::endl;
		}
	}


	namespace internal
	{
		void set_log_callback(void(*log_function)(const std::string_view& message))
		{
			s_log_function = log_function;
		}


		void resize_images_and_annotations(annotations::annotations_collection& collection, const std::pair<uint32_t, uint32_t>& desired_size, uint32_t desired_num_channels, const std::filesystem::path& target_folder, const std::optional<std::filesystem::path>& cache_folder)
		{
			const auto images_cache_folder = cache_folder == std::nullopt ? std::nullopt : std::make_optional(*cache_folder / (std::to_string(desired_size.first) + "_" + std::to_string(desired_size.second) + "_" +  std::to_string(desired_num_channels)));

			if(std::filesystem::exists(target_folder))
			{
				if(images_cache_folder.has_value())
				{
					// cache already generated images
					if(!std::filesystem::exists(*images_cache_folder))
					{
						std::filesystem::create_directories(*images_cache_folder);
					}
					for(const auto& v : std::filesystem::directory_iterator(target_folder))
					{
						const auto move_dest = (*images_cache_folder) / v.path().filename();
						if((v.path().extension() == ".png" || v.path().extension() == ".jpg") && !std::filesystem::exists(move_dest))
						{
							std::filesystem::rename(v.path(), move_dest);
						}
					}
				}

				// clear
				std::filesystem::remove_all(target_folder);
			}
			std::filesystem::create_directories(target_folder);

			std::vector<annotations::annotations> temp_collection;
			temp_collection.resize(collection.data.size());
			std::atomic_uint64_t last_update_secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			std::atomic_uint64_t local_num_image_done = 0;
			std::atomic_bool logged_progress_once = false;
			const uint32_t collection_size = collection.size();

			std::transform(std::execution::par_unseq, collection.begin(), collection.end(), temp_collection.begin(), [&last_update_secs, &local_num_image_done, &logged_progress_once, desired_size, desired_num_channels, target_folder, collection_size, images_cache_folder](const annotations::annotations& annotations)
			{
				annotations::annotations updated = annotations;

				const auto filename_img = annotations.filename_img.filename();
				const auto filename_txt = annotations.filename_txt.filename();

				// copy annotations
				std::filesystem::copy_file(annotations.filename_txt, target_folder / filename_txt);

				// transform and copy image
				// check cache first
				if(images_cache_folder.has_value() && std::filesystem::exists(*images_cache_folder / filename_img))
				{
					std::filesystem::copy_file(*images_cache_folder / filename_img, target_folder / filename_img);
				}
				else
				{
					// transform
					thread_local static std::vector<uint8_t> temp_data;
					const size_t img_size = desired_size.first * desired_size.second * desired_num_channels * 2;
					if(temp_data.size() < img_size)
					{
						temp_data.resize(desired_size.first * desired_size.second * desired_num_channels * 2);
					}

					std::string full_filename = annotations.filename_img.string();
					int w;
					int h;
					int channels;
					if(stbi_uc* image_data = stbi_load(full_filename.c_str(), &w, &h, &channels, (int)desired_num_channels))
					{
						if(w == (int)desired_size.first && h == (int)desired_size.second)
						{
							// just copy, no resizing needed
							std::filesystem::copy_file(annotations.filename_img, target_folder / filename_img);
						}
						else
						{
							const auto target_filename = (target_folder / filename_img).string();
							stbir_resize_uint8(image_data, w, h, 0, temp_data.data(), (int)desired_size.first, (int)desired_size.second, 0, (int)desired_num_channels);
							stbi_write_png(target_filename.c_str(), (int)desired_size.first, (int)desired_size.second, (int)desired_num_channels, temp_data.data(), 0);
						}
						stbi_image_free(image_data);
					}
					else
					{
						log("ERROR: Failed to load image: '" + full_filename + "'");
					}
				}

				// update 'annotations'
				updated.filename_img = target_folder / filename_img;
				updated.filename_txt = target_folder / filename_txt;

				// count
				local_num_image_done++;

				// report
				uint64_t now_secs = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
				const auto secs_since_last_update = now_secs - last_update_secs;
				if(secs_since_last_update > 2 || (local_num_image_done == collection_size && logged_progress_once))
				{
					last_update_secs = now_secs;
					log("Converting images... progress: " + std::to_string(local_num_image_done) + " / " + std::to_string(collection_size));
					logged_progress_once = true;
				}
				return updated;
			});
			collection.data = std::move(temp_collection);
			log("Converting images... done!");
		}

		bool write_yolo_data(const std::filesystem::path& dest_filepath, const yolo_data& data)
		{
			std::ofstream file;
			file.open (dest_filepath);
			if(!file.is_open())
			{
				return false;
			}

			file     <<         "classes = " << data.classes;
			file     << "\n" << "train = " << std::filesystem::canonical(std::filesystem::absolute(data.train)).string();
			file     << "\n" << "valid = " << std::filesystem::canonical(std::filesystem::absolute(data.valid)).string();
			file     << "\n" << "names = " << std::filesystem::canonical(std::filesystem::absolute(data.names)).string();
			if(data.backup.has_value())
			{
				file << "\n" << "backup = " << std::filesystem::canonical(std::filesystem::absolute(*data.backup)).string();
			}
			file.close();
			return true;
		}

		std::optional<std::filesystem::path> find_latest_backup_weights(const std::filesystem::path& folder_path)
		{
			std::optional<std::filesystem::path> latest;
			std::optional<std::filesystem::file_time_type> latest_write_time;
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
		static std::array<char, 128> str_to_c(const std::filesystem::path& str) { std::array<char, 128> v = {0}; strncpy(v.data(), str.c_str(), v.size()); return v; }

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
