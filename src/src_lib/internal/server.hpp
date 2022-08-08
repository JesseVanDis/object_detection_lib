#ifndef ALL_YOLO_SERVER_HPP
#define ALL_YOLO_SERVER_HPP

#include <filesystem>

namespace yolo::server
{
	struct init_args
	{
		std::filesystem::path images_and_txt_annotations_folder;
		std::filesystem::path weights_folder_path;
	};

	class server_internal
	{
		public:
			explicit server_internal(init_args&& init_args);

			bool is_running() const;

		private:
			const init_args m_init_args;
	};
}

#endif //ALL_YOLO_SERVER_HPP
