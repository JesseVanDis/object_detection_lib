#include <yolo.hpp>
#include <iostream>
#include <cstring>
#include <array>

namespace yolo::internal
{
	static std::optional<int> find_arg(int argc, const char** argv, const char *arg);
	static const char* find_arg_value(int argc, const char** argv, const char *arg);

	template<int NumValues>
	static std::optional<std::array<const char*, NumValues>> find_arg_values(int argc, const char** argv, const char *arg);

	void show_help()
	{
		std::cout << "Usage: " << std::endl;
		std::cout << "" << std::endl;
		std::cout << "	--fetch_from_open_images [class-name] [folder-path] [max-images]   " << std::endl;
		std::cout << "                                 downloads images and notations from google open images" << std::endl;
		std::cout << "                                 it will automatically format the annotation so that you" << std::endl;
		std::cout << "                                 can train it using '--train_yolov3 [dest]'" << std::endl;
		std::cout << "                                     class-name:  images will be filtered on this class name" << std::endl;
		std::cout << "                                                  examples: 'Cat', 'Dog', 'Chair' ect..." << std::endl;
		std::cout << "                                                  only 1 class-name supported so far" << std::endl;
		std::cout << "                                     folder-path: destination folder on which to store" << std::endl;
		std::cout << "                                                  the images and annotations" << std::endl;
		std::cout << "                                     max-images:  max amount of images to download" << std::endl;
		std::cout << "                                                  set to -1 for infinite" << std::endl;
		std::cout << "                                 example:" << std::endl;
		std::cout << "                                     --fetch_from_open_images Cat ./data 100" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "	--train_yolov3 [folder-path]   trains on the images/annotations that are in the given folder" << std::endl;
		std::cout << "                                 the folder content must look like this:" << std::endl;
		std::cout << "                                     FOLDER/img1.jpg, FOLDER/img1.txt, FOLDER/img2.jpg, FOLDER/img2.txt ect..." << std::endl;
		std::cout << "                                 the .txt annotation must be in YOLOv4 format" << std::endl;
		std::cout << "                                 the 'folder-path' can also be a server address, hosted by '--server'" << std::endl;
		std::cout << "                                 examples:" << std::endl;
		std::cout << "                                     --train_yolov3 ./data" << std::endl;
		std::cout << "                                     --train_yolov3 192.168.1.3:9090" << std::endl;
		std::cout << "" << std::endl;
		std::cout << "	--server [folder-path]         sets up a server for hosting images/annotations" << std::endl;
		std::cout << "                                 the folder content must look like this:" << std::endl;
		std::cout << "                                     FOLDER/img1.jpg, FOLDER/img1.txt, FOLDER/img2.jpg, FOLDER/img2.txt ect..." << std::endl;
		std::cout << "                                 when training happens while 'linked' to this server, the following will happen:" << std::endl;
		std::cout << "                                 ( 'trainer' will be referred here as 'the machine on which the training happens' )" << std::endl;
		std::cout << "                                     * If the server will share the images and annotations to the 'trainer'" << std::endl;
		std::cout << "                                     * If the server will share the weights file ( if it has any )" << std::endl;
		std::cout << "                                     * The 'trainer' will keep sharing the latest weights file back to the server again" << std::endl;
		std::cout << "                                 This is convenient when using this with google colab, where colab can disconnect the 'trainer' at any time." << std::endl;
		std::cout << "" << std::endl;
		std::cout << "  -h, --help                     shows this help" << std::endl;
		std::cout << "" << std::endl;
	}

	int tool_main(int argc, const char** argv)
	{
		if(argc < 2 || find_arg(argc, argv, "-h") || find_arg(argc, argv, "--help"))
		{
			show_help();
		}

		if(auto v = find_arg_values<3>(argc, argv, "--fetch_from_open_images"))
		{
			yolo::obtain_trainingdata_google_open_images(v->at(1), v->at(0), atoi(v->at(2)));
		}

		if(auto v = find_arg_values<1>(argc, argv, "--server"))
		{
			if(auto p_server = yolo::server::start(v->at(0)))
			{
				getchar(); // just wait for a key fow now. server will stay active until then.
			}
		}

		if(const char* folder = find_arg_value(argc, argv, "--train_yolov3"))
		{
			yolo::v3::train(folder);
		}

		//yolo::obtain_trainingdata_google_open_images("/home/jesse/MainSVN/catwatch_data/open_images", "Cat", 10000);
		//yolo::v3::train("/home/jesse/MainSVN/catwatch_data/open_images");


		return 0;

	}

	static std::optional<int> find_arg(int argc, const char** argv, const char *arg)
	{
		for(int i=0; i<argc; i++)
		{
			if(strcmp(argv[i], arg) == 0)
			{
				return i;
			}
		}
		return std::nullopt;
	}

	template<int NumValues>
	static std::optional<std::array<const char*, NumValues>> find_arg_values(int argc, const char** argv, const char *arg)
	{
		std::array<const char*, NumValues> v = {};
		if(auto k = find_arg(argc, argv, arg))
		{
			if((*k)+NumValues < argc)
			{
				for(int i=*k+1, j=0; i<std::min(argc, *k+(NumValues+1)); i++, j++)
				{
					v[j] = argv[i];
				}
			}
			return v;
		}
		return std::nullopt;
	}

	static const char* find_arg_value(int argc, const char** argv, const char* arg)
	{
		if(auto v = find_arg_values<1>(argc, argv, arg))
		{
			return (*v)[0];
		}
		return nullptr;
	}
}


int main(int argc, const char** argv)
{
	return yolo::internal::tool_main(argc, argv);
}

