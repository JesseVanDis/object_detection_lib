#ifndef ALL_YOLO_INTERNAL_HPP
#define ALL_YOLO_INTERNAL_HPP

#include <functional>
#include <optional>
#include "annotations.hpp"
#include "cfg.hpp"


namespace yolo::http
{
	struct progress;
}

namespace yolo::internal
{
	struct yolo_data
	{
		uint32_t 								classes;
		std::filesystem::path 					train;
		std::filesystem::path 					valid;
		std::filesystem::path 					names;
		std::optional<std::filesystem::path>	backup;
	};

	struct darknet_training_args
	{
		bool 	should_clear = false;
		bool 	dont_show = true;
		bool 	map = true;
		float 	thresh = 0.25f;
		float 	iou_thresh = 0.5f;
		int 	mjpeg_port = -1;
		bool 	show_imgs = false;
		bool 	benchmark_layers = false;
		std::optional<std::filesystem::path> chart_path = std::nullopt;
	};

	struct obtain_data_from_server_progress
	{
		size_t num_images_obtained;
		size_t total_num_images;
	};

	struct obtain_trainingdata_server_args
	{
		std::string server_and_port;
		std::filesystem::path dest_images_and_txt_annotations_folder;
		std::filesystem::path dest_weights_folder;
		std::optional<std::function<void(const obtain_data_from_server_progress& progress)>> progress_callback = std::nullopt;
		std::optional<std::function<void(const http::progress& progress)>> weights_progress_callback = std::nullopt;
		bool silent_images_and_txt_annotations = true;
		bool silent_weights = true;
	};

	void 									set_log_callback(void(*log_function)(const std::string_view& message));
	void 									resize_images_and_annotations(annotations::annotations_collection& collection, const std::pair<uint32_t, uint32_t>& desired_size, uint32_t desired_num_channels, const std::filesystem::path& target_folder, const std::optional<std::filesystem::path>& cache_folder = std::nullopt);
	bool 									write_yolo_data(const std::filesystem::path& dest_filepath, const yolo_data& data);
	std::optional<std::filesystem::path> 	find_latest_backup_weights(const std::filesystem::path& folder_path);
	bool 									start_darknet_training(const std::filesystem::path& model_cfg_data, const cfg::cfg& model_cfg, const std::filesystem::path& starting_weights, const darknet_training_args& args = {});
	std::optional<std::filesystem::path> 	obtain_starting_weights(const std::string& pretrained_weights_url, const std::optional<std::filesystem::path>& backup_path, const std::optional<std::filesystem::path>& download_target_path = std::nullopt);
	std::optional<std::filesystem::path> 	find_related_image_filepath(const std::filesystem::path& filepath_txt);
	std::optional<std::filesystem::path> 	find_latest_weights(const std::filesystem::path& base_folder_path);

	/// 									\param server_and_port for example example: "http://192.168.1.3:8086"
	/// 									\return false if it failed obtaining the data
	bool 									obtain_trainingdata_server(const obtain_trainingdata_server_args& args);
	bool 									obtain_trainingdata_server(const std::string_view& server_and_port, const std::filesystem::path& dest_images_and_txt_annotations_folder, const std::filesystem::path& dest_weights_folder);
}


#endif //ALL_YOLO_INTERNAL_HPP
