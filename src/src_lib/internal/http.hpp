#ifndef ALL_YOLO_HTTP_HPP
#define ALL_YOLO_HTTP_HPP

#include <string_view>
#include <filesystem>
#include <functional>

namespace yolo::http
{
	struct progress
	{
		size_t bytes_written;
		double progress_total;
		double progress_now;

		[[nodiscard]] double progress_perc() const { return progress_total > 0.0 ? (progress_now / progress_total) : -1.0; }
	};

	bool download(const std::string_view& url, const std::filesystem::path& dest_filepath, bool overwrite = false, const std::optional<std::function<void(const progress& progress)>>& progress_callback = std::nullopt);
	bool download(const std::string_view& url, std::vector<uint8_t>& dest, const std::optional<std::function<void(const progress& progress)>>& progress_callback = std::nullopt);
}


#endif //ALL_YOLO_HTTP_HPP
