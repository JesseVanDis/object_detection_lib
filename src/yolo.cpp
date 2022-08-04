#include <iostream>
#include <python3.8/Python.h>
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

		void train_on_colab(const std::filesystem::path& images_folder, const std::filesystem::path& weights_folder_path, const model_args& args)
		{

		}

		void obtain_trainingdata_google_open_images(const std::filesystem::path& target_images_folder, const std::optional<std::filesystem::path>& cache_folder)
		{
			/// https://storage.googleapis.com/openimages/web/download.html

			// https://www.tutorialspoint.com/how-to-create-a-virtual-environment-in-python

			// sudo apt install python3-virtualenv

			PyObject* pInt;

			Py_Initialize();

			std::stringstream python_command;

			python_command << std::endl << "import fiftyone as fo";
			python_command << std::endl << "import fiftyone.zoo as foz";
			python_command << std::endl << "";
			python_command << std::endl << "dataset = foz.load_zoo_dataset(";
			python_command << std::endl << "\"open-images-v6\",";
			python_command << std::endl << "		split=\"validation\",";
			python_command << std::endl << "		max_samples=100,";
			python_command << std::endl << "		seed=51,";
			python_command << std::endl << "		shuffle=True,";
			python_command << std::endl << ")";

			std::string python_command_str = python_command.str();
			const char* python_command_cstr = python_command_str.c_str();

			PyRun_SimpleString(python_command_cstr);

			Py_Finalize();


			// python3 -V

			// sudo apt install python3-pip
			// sudo apt install python3.8-venv
			// python3 -m venv tutorial-env
			// source tutorial-env/bin/activate
			// python -m pip install fiftyone


			/*
			 import fiftyone as fo
			import fiftyone.zoo as foz

			 dataset = foz.load_zoo_dataset(
				"open-images-v6",
				split="validation",
				max_samples=100,
				seed=51,
				shuffle=True,
			)
			 */

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
