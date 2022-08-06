#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_NO_TGA

//Werror=implicit-fallthrough

#include <stb_image.h>
#include <stb_image_resize.h>
#include <stb_image_write.h>


#if 0
namespace yolo
{
	void log(const std::string_view& message);

	std::optional<image> image::load(const std::filesystem::path& filepath)
	{
		image v;
		if(load(filepath, v))
		{
			return v;
		}
		return std::nullopt;
	}

	bool image::load(const std::filesystem::path& filepath, image& target)
	{

	}
}
#endif