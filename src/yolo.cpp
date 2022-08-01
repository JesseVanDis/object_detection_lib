#include <iostream>
#include "include/yolo.hpp"
#include "internal/annotations.hpp"
#include "internal/cfg.hpp"
#include "internal/internal.hpp"
#include "models/yolov3.h"
// https://colab.research.google.com/drive/1dT1xZ6tYClq4se4kOTen_u5MSHVHQ2hu

// print warning when 'max_batches' is > 10000, then it will need a lot of training to be done before a loss chart is generated. ( just for the user )

namespace yolo
{
	using namespace internal;

	void log(const std::string_view& message);

	namespace v3
	{
		struct full_args
		{
			uint32_t 						training_batch;
			uint32_t 						training_subdivisions;
			uint32_t 						training_max_batches;
			std::pair<uint32_t, uint32_t> 	image_size;
			uint32_t 						image_channels;
			uint32_t 						min_steps;
			uint32_t 						max_steps;
			uint32_t 						num_classes;
			uint32_t 						filters;
		};

		bool train(const std::filesystem::path& images_folder, const std::filesystem::path& weights_folder_path, const model_args& args)
		{
			const std::filesystem::path processed_dir = weights_folder_path / "tmp"; // std::filesystem::temp_directory_path();
			const std::filesystem::path cache_dir = weights_folder_path / "tmp" / "cache"; // std::filesystem::temp_directory_path();

			// load annotations
			auto all_set = annotations::annotations_collection::load(images_folder);
			if(!all_set.has_value())
			{
				log("Failed to load annotations");
				return false;
			}

			const uint32_t num_classes = all_set->num_classes();

			// setup model args
			const full_args model_arg = {
					.training_batch = args.training_batch,
					.training_subdivisions = args.training_subdivisions,
					.training_max_batches = args.training_max_batches,
					.image_size = args.image_size,
					.image_channels = args.image_channels,
					.min_steps = args.min_steps.value_or((uint32_t)((float)args.training_max_batches * 100.0f / 125.0f)),
					.max_steps = args.max_steps.value_or((uint32_t)((float)args.training_max_batches * 100.0f / 111.0f)),
					.num_classes = num_classes,
					.filters = (num_classes + 5) * 3 // or ((num_anchors/3)*(num_classes+5))
			};

			// load model cfg
			cfg::load_args cfg_load_args;
			cfg_load_args.predefinitions.emplace_back("training");
			cfg_load_args.variables.insert({"training_batch", 			std::to_string(model_arg.training_batch)});
			cfg_load_args.variables.insert({"training_subdivisions",	std::to_string(model_arg.training_subdivisions)});
			cfg_load_args.variables.insert({"training_max_batches", 	std::to_string(model_arg.training_max_batches)});
			cfg_load_args.variables.insert({"image_size_x", 			std::to_string(model_arg.image_size.first)});
			cfg_load_args.variables.insert({"image_size_y", 			std::to_string(model_arg.image_size.second)});
			cfg_load_args.variables.insert({"image_channels", 			std::to_string(model_arg.image_channels)});
			cfg_load_args.variables.insert({"min_steps", 				std::to_string(model_arg.min_steps)});
			cfg_load_args.variables.insert({"max_steps", 				std::to_string(model_arg.max_steps)});
			cfg_load_args.variables.insert({"num_classes", 				std::to_string(model_arg.num_classes)});
			cfg_load_args.variables.insert({"filters", 					std::to_string(model_arg.filters)});

			auto cfg = cfg::load(s_cfg_yolov3, cfg_load_args);
			if(!cfg.has_value())
			{
				log("Failed to load cfg file");
				return false;
			}

			// split into training / validate
			annotations::annotations_collection train_set;
			annotations::annotations_collection valid_set;
			all_set->split_to_training_and_valid_collections(train_set, valid_set, args.validation_ratio);

			// resize images / annotations
			const auto images_folder_resized_train = processed_dir / "train";
			const auto images_folder_resized_valid = processed_dir / "valid";

			resize_images_and_annotations(train_set, model_arg.image_size, model_arg.image_channels, images_folder_resized_train, cache_dir);
			resize_images_and_annotations(valid_set, model_arg.image_size, model_arg.image_channels, images_folder_resized_valid, cache_dir);

			// to darknet format
			const yolo_data yolo_data =
					{
							.classes = num_classes,
							.train = processed_dir / "train.txt",
							.valid = processed_dir / "val.txt",
							.names = processed_dir / "yolo.names",
							.backup = weights_folder_path
					};

			train_set.save_darknet_txt(yolo_data.train);
			valid_set.save_darknet_txt(yolo_data.valid);
			all_set->save_darknet_names(yolo_data.names);
			write_yolo_data(processed_dir / "yolo.data", yolo_data);

			// Obtain pretrained model
			const auto starting_weights_path = obtain_starting_weights("https://pjreddie.com/media/files/darknet53.conv.74", yolo_data.backup);
			if(!starting_weights_path.has_value())
			{
				log("Failed to obtain starting weights");
				return false;
			}

			start_darknet_training(processed_dir / "yolo.data", *cfg, *starting_weights_path);

			return true;
		}

		void offer_colab_training()
		{
			//todo
		}

		/// run YOLO v3 detection on an image
		//void detect(const std::filesystem::path& image, const std::filesystem::path& weights_filepath, const model_args& args)
		//{

		//}

		/// run YOLO v3 detection on an image
		//void detect(const image& image, const std::filesystem::path& weights_filepath, const model_args& args)
		//{

		//}
	}

}
