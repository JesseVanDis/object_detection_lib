#ifndef ALL_YOLO_INTERNAL_HPP
#define ALL_YOLO_INTERNAL_HPP

#include "annotations.hpp"
#include "cfg.hpp"

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

	void 									set_log_callback(void(*log_function)(const std::string_view& message));
	void 									resize_images_and_annotations(annotations::annotations_collection& collection, const std::pair<uint32_t, uint32_t>& desired_size, uint32_t desired_num_channels, const std::filesystem::path& target_folder, const std::optional<std::filesystem::path>& cache_folder = std::nullopt);
	bool 									write_yolo_data(const std::filesystem::path& dest_filepath, const yolo_data& data);
	std::optional<std::filesystem::path> 	find_latest_backup_weights(const std::filesystem::path& folder_path);
	bool 									start_darknet_training(const std::filesystem::path& model_cfg_data, const cfg::cfg& model_cfg, const std::filesystem::path& starting_weights, const darknet_training_args& args = {});
	std::optional<std::filesystem::path> 	obtain_starting_weights(const std::string& pretrained_weights_url, const std::optional<std::filesystem::path>& backup_path, const std::optional<std::filesystem::path>& download_target_path = std::nullopt);

}


#endif //ALL_YOLO_INTERNAL_HPP
