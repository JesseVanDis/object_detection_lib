#ifndef ALL_YOLO_ZIP_HPP
#define ALL_YOLO_ZIP_HPP

#include <filesystem>
#include <vector>

namespace yolo::zip
{
	bool create_zip_file(const std::filesystem::path& dest_filename, const std::vector<std::filesystem::path>& files_to_zip);
	bool extract_zip_file(const std::filesystem::path& zip_filename, const std::filesystem::path& dest_folder);
}

#endif //ALL_YOLO_ZIP_HPP
