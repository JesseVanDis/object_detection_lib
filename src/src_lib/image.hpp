#ifndef ALL_YOLO_IMAGE_HPP
#define ALL_YOLO_IMAGE_HPP

#if 0

#include <cstdint>
#include <vector>
#include <filesystem>
#include <memory>

namespace yolo
{
	enum class image_format
	{
		rgb,
		count
	};

	struct image
	{
		uint32_t 					width_px = 0u;
		uint32_t 					height_px = 0u;
		std::vector<uint8_t> 		data;
		image_format 				format =  image_format::count;

		static std::optional<image> load(const std::filesystem::path& filepath);
		static bool load(const std::filesystem::path& filepath, image& target);

	};
}

#endif
#endif //ALL_YOLO_IMAGE_HPP
