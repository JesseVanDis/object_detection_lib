#ifndef YOLO_ALL_LIBRARY_HPP
#define YOLO_ALL_LIBRARY_HPP

#include <cstdint>
#include <optional>
#include <filesystem>
#include <functional>
#include <tuple>

namespace yolo
{
	struct image;

	namespace internal
	{
		struct folder_and_server
		{
			std::optional<std::string> server;
			std::filesystem::path images_and_txt_annotations_folder;
			std::filesystem::path weights_folder_path = "./weights";
		};
	}

	namespace http::server
	{
		class server_internal;
		class server
		{
			protected:
				friend std::unique_ptr<server> start(const std::filesystem::path& images_and_txt_annotations_folder, const std::filesystem::path& weights_folder_path, const std::filesystem::path& chart_png_path, const std::optional<std::filesystem::path>& latest_weights_filepath, unsigned int port);
				explicit server(std::unique_ptr<server_internal>&& v);
				const std::unique_ptr<server_internal> m_internal;
			public:
				~server();

				enum
				{
					DEFAULT_PORT = 8086
				};
		};

		/// sets up a server for hosting images/annotations
		/// the folder content must look like this:
		///     FOLDER/img1.jpg, FOLDER/img1.txt, FOLDER/img2.jpg, FOLDER/img2.txt ect...
		/// when training happens while 'linked' to this server, the following will happen:
		/// ( 'trainer' will be referred here as 'the machine on which the training happens' )
		///     * If the server will share the images and annotations to the 'trainer'
		///     * If the server will share the weights file ( if it has any )
		///     * The 'trainer' will keep sharing the latest weights file back to the server again
		/// This is convenient when using this with google colab, where colab can disconnect the 'trainer' at any time.
		/// The server will close upon destruction of the returning object.
		///
		/// \param images_and_txt_annotations_folder folder with the images and annotations (.txt in YOLOv4 format) to train on.
		///                                          The data inside the folder must be structured like so: 'img_1.jpg, img_1.txt, img_2.jpg, img_2.txt'. So no subdirectories, and the name of the jpg and txt must match.
		///                                          It will automatically split into 'training' and 'eval' sections.
		/// \param weights_folder_path
		/// \return nullptr or server object. If null, the starting of the server failed. If not null, server is up and will close upon destruction of this object.
		std::unique_ptr<server> start(const std::filesystem::path& images_and_txt_annotations_folder, const std::filesystem::path& weights_folder_path = "./weights", const std::filesystem::path& chart_png_path = "./chart.png", const std::optional<std::filesystem::path>& latest_weights_filepath = std::nullopt, unsigned int port = server::DEFAULT_PORT);
	}

	namespace v3
	{
		/// usefull link: https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1
		struct model_args // NOLINT
		{
			uint32_t training_batch = 64;

			uint32_t training_subdivisions = 16;

			/// amount of batches to do before signalling it is done with the training
			uint32_t training_max_batches = 6000;

			/// All images and its annotation files will be internally rescaled to the given size. If an image is already at that size, nothing will be done to save performance.
			std::pair<uint32_t, uint32_t> image_size = {512, 512};

			/// All images should have the amount of channels given here. No internal conversion is done *yet*
			uint32_t image_channels = 3;

			/// min_steps default = max_batches*0.8
			std::optional<uint32_t> min_steps = std::nullopt;

			/// max_steps default = max_batches*0.9
			std::optional<uint32_t> max_steps = std::nullopt;

			// #5% (small dataset) to 30% (large dataset)
			float validation_ratio = 0.05;
		};

		/// Train YOLO v3 on a dataset.
		/// \param images_and_txt_annotations_folder folder with the images and annotations (.txt in YOLOv4 format) to train on.
		///                                          The data inside the folder must be structured like so: 'img_1.jpg, img_1.txt, img_2.jpg, img_2.txt'. So no subdirectories, and the name of the jpg and txt must match.
		///                                          It will automatically split into 'training' and 'eval' sections.
		///
		///                                          A YOLO .txt annotation format example (class_id, x, y, width, height):
		///                                              0 0.658 0.696 0.079 0.141
		///                                              0 0.712 0.688 0.095 0.119
		///
		/// \param trained_model_dest_filepath target folder or filepath ( example: weights.data )
		///
		/// \param args          yolo v3 model arguments.
		bool train(const std::filesystem::path& images_and_txt_annotations_folder, const std::filesystem::path& weights_folder_path = "./weights", const model_args& args = {});

		/// same as 'train', but prints a message of instructions on how to do so on google colab, which offers good GPU's
		/// would be cool if this could be automated trough an API or something...
		/// visit: https://colab.research.google.com/github/JesseVanDis/object_detection_lib/blob/main/train.ipynb
		void train_on_colab(const std::filesystem::path& images_and_txt_annotations_folder, const std::filesystem::path& weights_folder_path = "./weights", const std::filesystem::path& chart_png_path = "./chart.png", const std::optional<std::filesystem::path>& latest_weights_filepath = std::nullopt, const model_args& args = {}, unsigned int port = http::server::server::DEFAULT_PORT);

		/// run YOLO v3 detection on an image
		//void detect(const std::filesystem::path& image, const std::filesystem::path& weights_filepath = "./trained.weights", const model_args& args = {});

		/// run YOLO v3 detection on an image
		//void detect(const image& image, const std::filesystem::path& weights_filepath = "./trained.weights", const model_args& args = {});
	}

	/// pull training data from a server ( with --server )
	/// \param server example: "http://192.168.1.3:8086"
	/// \return returns the path to which it downloaded the data. or nullopt of the downloading failed
	std::optional<internal::folder_and_server> obtain_trainingdata_server(const std::string_view& server);

	/// downloads an existing dataset from google open images, with the matched tags. It will use the opensource tool FiftyOne
	/// \param target_images_folder Target folder to save the images and notation files to. Format will be YOLOv4 (img1.jpg, img1.txt, img2.jpg, img2.txt ect... no sub-folders)
	/// \param class_name class name of the images that should be downloaded. examples: "Cat", "Dog", "Human" ect...  Combining not possible (yet)
	/// \param max_samples
	void obtain_trainingdata_google_open_images(const std::filesystem::path& target_images_folder, const std::string_view& class_name, const std::optional<size_t>& max_samples);

	/// downloads an existing dataset from google open images, with the matched tags. It will use the opensource tool FiftyOne
	/// \param target_images_folder Target folder to save the images and notation files to. Format will be YOLOv4 (img1.jpg, img1.txt, img2.jpg, img2.txt ect... no sub-folders)
	/// \param query self made up query for describing what content to fetch
	///              looks like this: open_images,[subject],[max_samples]
	//               example: open_images,cat,5000
	void obtain_trainingdata_google_open_images(const std::filesystem::path& target_images_folder, const std::string_view& query);

	/// sets log callback for the given function. If not set, it will use std::cout
	void set_log_callback(void(*log_function)(const std::string_view& message));
}


#endif //YOLO_ALL_LIBRARY_HPP
