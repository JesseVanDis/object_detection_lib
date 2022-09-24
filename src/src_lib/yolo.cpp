#include <iostream>
#include <yolo.hpp>
#include <fstream>
#include "internal/annotations.hpp"
#include "internal/cfg.hpp"
#include "internal/internal.hpp"
#include "internal/python.hpp"
#include "internal/http.hpp"
#include "internal/http_server.hpp"
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

		bool train(const std::filesystem::path& images_and_txt_annotations_folder, const std::filesystem::path& weights_folder_path, const model_args& args)
		{
			const std::filesystem::path processed_dir = weights_folder_path / "tmp"; // std::filesystem::temp_directory_path();
			const std::filesystem::path cache_dir = weights_folder_path / "tmp" / "cache"; // std::filesystem::temp_directory_path();

			// load annotations
			auto all_set = annotations::annotations_collection::load(images_and_txt_annotations_folder);
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

			// to darknet format
			const yolo_data yolo_data =
					{
							.classes = num_classes,
							.train = processed_dir / "train.txt",
							.valid = processed_dir / "val.txt",
							.names = processed_dir / "yolo.names",
							.backup = weights_folder_path
					};

			if(!train_set.save_darknet_txt(yolo_data.train))
			{
				log("Failed to write '" + yolo_data.train.string() + "'");
				return false;
			}
			if(!valid_set.save_darknet_txt(yolo_data.valid))
			{
				log("Failed to write '" + yolo_data.valid.string() + "'");
				return false;
			}
			if(!all_set->save_darknet_names(yolo_data.names))
			{
				log("Failed to write '" + yolo_data.names.string() + "'");
				return false;
			}
			if(!write_yolo_data(processed_dir / "yolo.data", yolo_data))
			{
				log("Failed to write 'yolo.data' to '" + (processed_dir / "yolo.data").string() + "'");
				return false;
			}

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

		void train_on_colab(const std::string_view& data_source, const std::filesystem::path& weights_folder_path, const std::filesystem::path& chart_png_path, const std::optional<std::filesystem::path>& latest_weights_filepath, const model_args& args, unsigned int port)
		{
			auto public_ip = yolo::http::fetch_public_ipv4();
			log("-------------------------");
			log("Please open the following link: ");
			log("https://colab.research.google.com/github/JesseVanDis/object_detection_lib/blob/main/train.ipynb");
			log("And run the notebook.");
			log("You can leave the 'source' field empty.");
			if(public_ip.has_value())
			{
				log("    ( It should default to this machine: '" + *public_ip + ":" + std::to_string(port) + "' )");
			}
			log("-------------------------");
			log("");

			auto p_server = yolo::http::server::start(data_source, weights_folder_path, chart_png_path, latest_weights_filepath, port);
			if(p_server == nullptr)
			{
				log("Error: Server failed to start.");
				return;
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			if(public_ip.has_value())
			{
				const std::string test_url = "http://" + *public_ip + ":" + std::to_string(port) + "/test";
				log("testing '" + test_url + "'...");
				if(auto test_result = http::download_str(test_url))
				{
					log("testing '" + test_url + "'... Ok!");
					log("server can be contacted at '" + *public_ip + ":" + std::to_string(port) + "'. Everything is set on this side. Please continue to the colab page");
				}
				else
				{
					log("testing '" + test_url + "'... Failed");
					log("Could not verify that the server is running.");
					log("you may have port forwarded 8080 to a different port ( which is fine, but then change the port number in the 'source' field in colab as well )");
					log("or, you still need to set up the port forwarding in your router.");
					log("or, everything ok, and you ISP does not allow you to make calls to your own public ip address ( yes, that happens ). In that case, just try you the colab link");
				}
			}

			(void)args;

			getchar(); // just wait for a key for now. server will stay active until then.
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

	std::optional<internal::folder_and_server> obtain_trainingdata_server(const std::string_view& server)
	{
		if(server.empty())
		{
			return std::nullopt;
		}
		if(server.starts_with("http"))
		{
			const std::filesystem::path images_and_txt_annotations_folder = std::filesystem::temp_directory_path() / "data_from_server";
			const std::filesystem::path weights_folder = "./weights";
			if(internal::obtain_trainingdata_server(server, images_and_txt_annotations_folder, weights_folder))
			{
				return internal::folder_and_server{std::string(server), images_and_txt_annotations_folder, weights_folder};
			}
			return std::nullopt;
		}
		if(server.starts_with("open_images,"))
		{
			const std::filesystem::path images_and_txt_annotations_folder = std::filesystem::temp_directory_path() / "data_from_open_images";
			yolo::obtain_trainingdata_google_open_images(images_and_txt_annotations_folder, server);
			return internal::folder_and_server{std::nullopt, images_and_txt_annotations_folder}; // data is saved to 'images_and_txt_annotations_folder' so just use it as a folder ( not a server )
		}
		return internal::folder_and_server{std::nullopt, server}; // not an url. server = folder
	}


	bool obtain_trainingdata_google_open_images(const std::filesystem::path& target_images_folder, const std::string_view& class_name, const std::optional<size_t>& max_samples)
	{
		std::string query = "open_images," + std::string(class_name) + (max_samples.has_value() ? ("," + std::to_string(*max_samples)) : std::string(""));
		return obtain_trainingdata_google_open_images(target_images_folder, query);
	}

	struct QueryParams
	{
		std::string class_name;
		std::optional<size_t> max_samples;
	};

	static std::optional<QueryParams> parse(const std::string_view& query)
	{
		QueryParams retval;
		std::string word;
		size_t word_index = 0;
		for(size_t i=0; i<=query.size(); i++)
		{
			const char c = i == query.size() ? ',' : query[i];
			if(c != ',')
			{
				word += c;
			}
			else
			{
				char* word_cstr = const_cast<char*>(word.c_str());
				if(word_index != 0)
				{
					if(word_index == 1)
					{
						retval.class_name = word;
					}
					else if(word_index == 2)
					{
						const size_t amount = std::strtoul(word_cstr, &word_cstr, 10);
						if(*word_cstr != '\0')
						{
							log("failed to parse number in '" + std::string(query) + "'");
							return std::nullopt;
						}
						retval.max_samples = amount;
					}
				}
				if(word_index == 0 && word != "open_images")
				{
					log("query does not start with 'open_images'. please set the query as 'open_images,arg1,arg2,arg3' ect... in '" + std::string(query) + "'");
					return std::nullopt;
				}
				word_index++;
				word.clear();
			}
		}
		if(retval.class_name.empty())
		{
			return std::nullopt;
		}
		return retval;
	}

	bool obtain_trainingdata_google_open_images(const std::filesystem::path& target_images_folder, const std::string_view& query)
	{
		// uncomment the below to use the internal interpreter
		//#ifdef PYTHON3_FOUND
		//	#define OBTAIN_USE_INTERNAL_PYTHON
		//#endif

		/// https://storage.googleapis.com/openimages/web/download.html

		// create folder structure
		const auto temp_target_folder = std::filesystem::path(target_images_folder.string() + "_tmp");

		if(std::filesystem::exists(target_images_folder))
		{
			std::filesystem::remove(target_images_folder);
		}
		if(!std::filesystem::exists(temp_target_folder))
		{
			std::filesystem::create_directories(temp_target_folder);
		}

		// parse query
		std::optional<QueryParams> query_params = parse(query);
		if(!query_params)
		{
			log("Failed to parse open_images query");
			return false;
		}

		// python stuff...
		std::vector<std::string> py;
		//std::stringstream py;
		py.emplace_back("print(\"opening 'fiftyone' module...\");");
		py.emplace_back("");
		py.emplace_back("import fiftyone as fo");
		py.emplace_back("import fiftyone.zoo as foz");
		py.emplace_back("from fiftyone import ViewField as F");
		py.emplace_back("");
		py.emplace_back("print('Downloading open images dataset...')");
		py.emplace_back("dataset = foz.load_zoo_dataset(");
		py.emplace_back("		\"open-images-v6\","); // supported databases: https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html
		//py.emplace_back("		split=\"validation\","); // comment this to get more images, or uncomment is for fast testing
		py.emplace_back("		label_types=[\"detections\"],");
		py.emplace_back("		label_field=\"ground_truth\",");
		py.emplace_back("		classes=[\"" + query_params->class_name + "\"],");
		if(query_params->max_samples)
		{
			py.emplace_back("		max_samples=" + std::to_string(*query_params->max_samples) + ",");
		}
		py.emplace_back("		seed=51,");
		py.emplace_back("		shuffle=True,");
		py.emplace_back(")");
		py.emplace_back("");
		py.emplace_back("dataset_filtered = dataset.filter_labels(\"ground_truth_detections\", F(\"label\")==\"" + query_params->class_name + "\").filter_labels(\"ground_truth_detections\", F(\"IsDepiction\")==False)"); // NOLINT
		py.emplace_back("");
		//py 	<< "for sample in dataset_filtered:"; // uncomment for help in filtering
		//py 	<< "	print(sample)";
		//py 	<< "";
		//py 	<< "session = fo.launch_app(dataset_filtered)"; // uncomment for preview
		//py 	<< "time.sleep(9999999)"; // uncomment for preview
		py.emplace_back("print('Exporting dataset...')");
		py.emplace_back("dataset_filtered.export(");
		py.emplace_back("		export_dir=\"" + temp_target_folder.string() + "\",");
		py.emplace_back("		dataset_type=fo.types.YOLOv4Dataset,");
		py.emplace_back("		label_field=\"ground_truth_detections\",");
		py.emplace_back(")");

		#ifndef OBTAIN_USE_INTERNAL_PYTHON
		{
			// create a temporary python script
			const std::filesystem::path script_path = std::filesystem::temp_directory_path() / "temp.py";
			const std::string script_path_str = script_path.string();
			std::ofstream file;
			file.open (script_path_str.c_str());
			if(!file.is_open())
			{
				log("Failed to open file '" + script_path_str + "'");
				return false;
			}
			for(auto& line : py)
			{
				file << line << "\n";
			}
			file.close();
			std::string make_executable_cmd = "chmod +x \"" +  script_path_str + "\"";
			std::string run_cmd = "python3 \"" +  script_path_str + "\"";
			int res = 0;
			log("Running '" + make_executable_cmd + "'...");
			res = system(make_executable_cmd.c_str());
			if(res != 0)
			{
				log("Running '" + make_executable_cmd + "'... Failed. Exitted with code: " + std::to_string(res));
				return false;
			}
			log("Running '" + run_cmd + "'...");
			res = system(run_cmd.c_str());
			if(res != 0)
			{
				log("Running '" + make_executable_cmd + "'... Failed. Exitted with code: " + std::to_string(res));
				return false;
			}
		}
		#else
		{
			python::init_args init = { .print_callback = [](const std::string_view& str) { log(str); } };

			log("Starting python session...");
			auto py_instance = python::new_instance(init);
			{
				auto builder = py_instance->code_builder();
				for(auto& line : py)
				{
					builder << line;
				}

				builder.run();
			}
		}
		#endif

		if(std::filesystem::exists(temp_target_folder / "data"))
		{
			std::filesystem::rename(temp_target_folder / "data", target_images_folder);
			std::filesystem::rename(temp_target_folder / "obj.names", target_images_folder / "obj.names");
			std::filesystem::remove_all(temp_target_folder);
		}
		else
		{
			log("Failed to obtain from open images");
			return false;
		}

		return true;
		#undef OBTAIN_USE_INTERNAL_PYTHON
	}

	namespace http::server
	{
		std::unique_ptr<server> start(const std::string_view& data_source, const std::filesystem::path& weights_folder_path, const std::filesystem::path& chart_png_path, const std::optional<std::filesystem::path>& latest_weights_filepath, unsigned int port)
		{
#ifdef MINIZIP_FOUND
			yolo::http::server::init_args args = {
					.data_source = std::string(data_source),
					.weights_folder_path = weights_folder_path,
					.chart_png_path = chart_png_path,
					.latest_weights_filepath = latest_weights_filepath,
					.port = port
			};

			std::unique_ptr<server_internal> t = std::make_unique<server_internal>(std::move(args));
			if(!t->is_running())
			{
				return nullptr;
			}
			return std::unique_ptr<server>(new server(std::move(t)));
#else
			(void)images_and_txt_annotations_folder;
			(void)weights_folder_path;
			log("This library was build without minizip. 'server::start' cannot be used");
			return nullptr;
#endif
		}
	}
}
