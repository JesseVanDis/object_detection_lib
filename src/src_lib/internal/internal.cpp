#define STBI_NO_TGA

#include <iostream>
#include <fstream>
#include <darknet.h>
#include <thread>
#include <map>
#include <yolo.hpp>
#include "internal.hpp"
#include "http.hpp"
#include "zip.hpp"
#include "option_list.h"
#include "data.h"
#include "demo.h"
#include "utils.h"

#ifdef GPU_SHOW_INFO
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#endif

static constexpr size_t s_max_images_per_batch = 100;
static constexpr size_t s_str_to_c_size = 128;

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

					const http::additional_download_args download_args = {
							.overwrite = true,
							.progress_callback = download_update
					};

					if(!yolo::http::download(pretrained_weights_url, pretrained_model_path, download_args))
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

		static void find_latest_weights(const std::filesystem::path& base_folder_path, std::filesystem::file_time_type& latest_write, std::optional<std::filesystem::path>& latest_path) // NOLINT
		{
			if(!std::filesystem::exists(base_folder_path) || !std::filesystem::is_directory(base_folder_path))
			{
				return;
			}
			for (const auto& path_it : std::filesystem::directory_iterator(base_folder_path))
			{
				const auto& path = path_it.path();
				if(std::filesystem::is_directory(path))
				{
					find_latest_weights(path, latest_write, latest_path);
				}
				else if (path.extension() == ".weights")
				{
					const auto write_time = std::filesystem::last_write_time(path);
					if(latest_path == std::nullopt || write_time > latest_write)
					{
						latest_write = write_time;
						latest_path = path;
					}
				}
			}
		}

		std::optional<std::filesystem::path> find_latest_weights(const std::filesystem::path& base_folder_path)
		{
			std::filesystem::file_time_type latest_write;
			std::optional<std::filesystem::path> latest_path;
			find_latest_weights(base_folder_path, latest_write, latest_path);
			return latest_path;
		}

		static std::array<char, s_str_to_c_size> str_to_c(const std::filesystem::path& str)
		{
			std::array<char, s_str_to_c_size> v = {0};
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

		static bool str_to_c(const std::filesystem::path& str, char* dest, size_t dest_capacity)
		{
			auto arr = str_to_c(str);
			memcpy(dest, arr.data(), std::min(dest_capacity, arr.size()));
			return str.empty() || arr[0] != '\0';
		}

		struct init_darknet_result
		{
			int gpu_index = -1;
			int *gpus = nullptr;
			int ngpus = 0;
		};

		static init_darknet_result init_darknet();

		//static std::filesystem::path save(const cfg::cfg& model_cfg, const std::filesystem::path& path)
		//{;
		//	model_cfg.save(path);
		//	path;
		//}

		static std::filesystem::path generate_unique_temp_filename(const std::string& base_name, const std::string& extension)
		{
			const uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
			static uint64_t s_uid = now;
			s_uid += now;
			const uint64_t uid = (s_uid + now) & 0xffffffff;
			char filename[255];
			snprintf(filename, sizeof(filename)-1, "%s_%#010x.%s", base_name.c_str(), (int)uid, extension.c_str());
			const std::filesystem::path tmp_path = std::filesystem::temp_directory_path();
			return tmp_path / filename;
		}

		struct common_darknet_args
		{
				common_darknet_args(const std::filesystem::path& model_cfg_data, const cfg::cfg& model_cfg, const std::filesystem::path& weights)
				{
					str_to_c(model_cfg_data, this->model_cfg_data, sizeof(this->model_cfg_data));
					auto p = generate_unique_temp_filename("model", "cfg");
					model_cfg.save(p);
					str_to_c(p, this->model_cfg, sizeof(this->model_cfg));
					str_to_c(weights, this->weights, sizeof(this->weights));
				}

				char 	model_cfg_data[128];
				char 	model_cfg[128];
				char 	weights[128];
		};

		bool start_darknet_training(const std::filesystem::path& model_cfg_data, const cfg::cfg& model_cfg, const std::filesystem::path& starting_weights, const darknet_training_args& args)
		{
			common_darknet_args base_args(model_cfg_data, model_cfg, starting_weights);

			auto darknet = init_darknet();

			// !./darknet detector train data/yolo.data cfg/yolov3_custom_train.cfg darknet53.conv.74 -dont_show -mjpeg_port 8090 -map

			auto chart_path_c = str_to_c(args.chart_path.has_value() ? std::filesystem::weakly_canonical(std::filesystem::absolute(*args.chart_path)) : "");

			train_detector(
					base_args.model_cfg_data,
					base_args.model_cfg,
					base_args.weights,
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

		bool start_darknet_demo(const cfg::cfg& model_cfg, const std::filesystem::path& weights, const std::filesystem::path& source, const darknet_demo_args& args)
		{
			common_darknet_args base_args("", model_cfg, weights);

			uint32_t classes = 0;
			model_cfg.get_value("", "classes", classes);
			if(classes == 0)
			{
				return false; // we can maybe also just set it to 20 ? ( like in the darknet sample, which is set by default )
			}

			// names ( the darknet way )
			char** names = nullptr;
			{
				std::unordered_map<int, std::string> names_map;
				if(args.names != std::nullopt)
				{
					names_map = *args.names;
				}
				else
				{
					names_map.insert({0, "."});
				}
				names = (char**)xcalloc(names_map.size(), sizeof(void*));
				for(const auto& v : names_map)
				{
					char* name = (char*)xmalloc((v.second.size()+1) * sizeof(char));
					strncpy(name, v.second.c_str(), (v.second.size()+1) * sizeof(char));
					names[v.first] = name;
				}
			}

			auto in_filename = str_to_c(source);
			auto out_filename = str_to_c(args.out_filename.has_value() ? std::filesystem::weakly_canonical(std::filesystem::absolute(*args.out_filename)) : "");
			auto http_post_host = str_to_c(args.http_post_host.has_value() ? std::filesystem::weakly_canonical(std::filesystem::absolute(*args.http_post_host)) : "");
			auto prefix = str_to_c(args.prefix);

			int cam_index = 0;
			const std::string source_str = source.string();
			if(source_str.starts_with("/dev/video"))
			{
				std::string substr = source_str.substr(strlen("/dev/video"));
				char *endptr;
				cam_index = (int)strtol(substr.c_str(), &endptr, 10);
				if (*endptr != '\0')
				{
					log("Could not obtain camera index from '" + source_str + "'");
					return false;
				}
				in_filename[0] = '\0';
			}
			else
			{
				in_filename = str_to_c(std::filesystem::weakly_canonical(std::filesystem::absolute(source)));
			}
			cam_index = 0;

			demo(
					base_args.model_cfg,
					(char*)"/home/jesse/MainSVN/catwatch/v2/src/cmake-build-debug/object_detection_lib/src/session_1664043901/weights/model_1860.weights",
					args.thresh,
					args.hier_thresh,
					cam_index,
					in_filename[0] == '\0' ? nullptr : in_filename.data(),
					names,
					(int)classes,
					args.avgframes,
					args.frame_skip,
					prefix.data(),
					out_filename[0] == '\0' ? nullptr : out_filename.data(),
				 	args.mjpeg_port,
					args.dontdraw_bbox ? 1 : 0,
					args.json_port,
					args.benchmark ? 0 : args.dont_show,
					args.ext_output ? 1 : 0,
					args.letter_box ? 1 : 0,
					args.time_limit_sec,
					http_post_host[0] == '\0' ? nullptr : http_post_host.data(),
					args.benchmark ? 1 : 0,
					args.benchmark_layers);

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

		static std::optional<std::map<unsigned int, std::string>> parse_images_list(const std::vector<uint8_t>& get_data_source_data)
		{
			std::map<unsigned int, std::string> map;

			std::string line;
			unsigned int index = ~0u;
			std::string image_name;
			size_t start = 0;

			// ignore the first line ( which just describer what source type it is )
			while(get_data_source_data[start] != '\0' && get_data_source_data[start] != '\n')
			{
				start++;
			}
			if(get_data_source_data[start] != '\0')
			{
				start++;
			}

			// parse the list
			for(size_t i=start; i<get_data_source_data.size(); i++)
			{
				const auto& v = get_data_source_data[i];
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

		static bool obtain_data_from_open_images(const obtain_trainingdata_server_args& args, const std::vector<uint8_t>& get_data_source_data)
		{
			std::string open_images_query = strchr((const char*)get_data_source_data.data(), '\n')+1;
			std::replace(open_images_query.begin(), open_images_query.end(), '\n', '\0');
			open_images_query.resize(strlen(open_images_query.c_str()));
			if(open_images_query.size() < strlen("open_images"))
			{
				return false;
			}
			return yolo::obtain_trainingdata_google_open_images(args.dest_images_and_txt_annotations_folder, open_images_query);
		}

		static bool obtain_data_from_images_list(const obtain_trainingdata_server_args& args, const std::vector<uint8_t>& get_data_source_data)
		{
			auto images_list = parse_images_list(get_data_source_data);
			if(!images_list.has_value() || images_list->empty())
			{
				return false;
			}

			// obtain missing images
			std::vector<unsigned int> missing_images;
			{
				for(const auto& v : *images_list)
				{
					if(!std::filesystem::exists(args.dest_images_and_txt_annotations_folder / (v.second + ".txt")))
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
					const std::filesystem::path dest_file = args.dest_images_and_txt_annotations_folder / filename;
					const http::additional_download_args download_args = {
							.overwrite = true,
							.silent = args.silent_images_and_txt_annotations
					};
					http::download(args.server_and_port + "/get_images?from=" + std::to_string(v.first) + "&to=" + std::to_string(v.second), dest_file, download_args);
					if(std::filesystem::exists(dest_file))
					{
						zip::extract_zip_file(dest_file, args.dest_images_and_txt_annotations_folder);
						std::filesystem::remove(dest_file);
					}
					num_images_done += (v.second - v.first)+1;

					obtain_data_from_server_progress progress = {
							.num_images_obtained = num_images_done,
							.total_num_images = total_num_images
					};
					if(args.progress_callback != std::nullopt)
					{
						(*args.progress_callback)(progress);
					}
				}
			}
			return true;
		}

		bool obtain_trainingdata_server(const obtain_trainingdata_server_args& args)
		{
			// download the latest weights
			{
				const std::filesystem::path dest_file = std::filesystem::temp_directory_path() / "weights_tmp.zip";
				const http::additional_download_args download_args =
						{
								.overwrite = true,
								.silent = args.silent_weights,
								.progress_callback = args.weights_progress_callback,
								.on_error = [](const char*, int http_code)
								{
									if(http_code == 204)
									{
										log("server does not have any weights to continue on. a fresh start will be created.");
										return true;
									}
									return false;
								}
						};
				http::download(args.server_and_port + "/latest_weights", dest_file, download_args);
				if(std::filesystem::exists(dest_file))
				{
					if(!std::filesystem::exists(args.dest_weights_folder))
					{
						std::filesystem::create_directories(args.dest_weights_folder);
					}
					zip::extract_zip_file(dest_file, args.dest_weights_folder);
					std::filesystem::remove(dest_file);
				}
			}

			// download images
			if(!std::filesystem::exists(args.dest_images_and_txt_annotations_folder))
			{
				std::filesystem::create_directories(args.dest_images_and_txt_annotations_folder);
			}

			std::vector<uint8_t> get_data_source_data;
			if(!http::download(std::string(args.server_and_port) + "/get_data_source", get_data_source_data))
			{
				return false;
			}

			std::string data_source_type = (const char*)get_data_source_data.data();
			{
				std::replace(data_source_type.begin(), data_source_type.end(), '\n', '\0');
				data_source_type.resize(strlen(data_source_type.c_str()));
			}

			if(data_source_type == "images_list" && !obtain_data_from_images_list(args, get_data_source_data))
			{
				return false;
			}
			else if(data_source_type == "open_images" && !obtain_data_from_open_images(args, get_data_source_data))
			{
				return false;
			}

			return true;
		}

		bool obtain_trainingdata_server(const std::string_view& server_and_port, const std::filesystem::path& dest_images_and_txt_annotations_folder, const std::filesystem::path& dest_weights_folder)
		{
			auto last_status_update = std::chrono::system_clock::now();
			log("downloading data from '" + std::string(server_and_port) + "'...");

			auto progress_callback = [&](const obtain_data_from_server_progress& progress)
			{
				const auto now = std::chrono::system_clock::now();
				if(now - last_status_update > std::chrono::seconds(1))
				{
					log("downloading data from '" + std::string(server_and_port) + "'... [" + std::to_string(progress.num_images_obtained) + "\\" + std::to_string(progress.total_num_images) + "]");
				}
			};

			const obtain_trainingdata_server_args args =
					{
					.server_and_port = std::string(server_and_port),
					.dest_images_and_txt_annotations_folder = dest_images_and_txt_annotations_folder,
					.dest_weights_folder = dest_weights_folder,
					.progress_callback = progress_callback,
					.weights_progress_callback = std::nullopt,
					.silent_images_and_txt_annotations = true,
					.silent_weights = false
			};

			const bool ok = obtain_trainingdata_server(args);
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
