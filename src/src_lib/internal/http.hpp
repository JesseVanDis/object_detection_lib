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

	bool download(const std::string_view& url, const std::filesystem::path& dest_filepath, bool overwrite = false, const std::optional<std::function<void(const progress& progress)>>& progress_callback = std::nullopt, bool silent = false);
	bool download(const std::string_view& url, std::vector<uint8_t>& dest, const std::optional<std::function<void(const progress& progress)>>& progress_callback = std::nullopt, bool silent = false);
	bool upload(const std::string_view& url, const std::filesystem::path& file_to_upload, const std::string_view& name = "data", bool silent = false);
	std::optional<std::string> download_str(const std::string_view& url, bool silent = true);
	std::optional<std::string> fetch_public_ipv4();
}


#endif //ALL_YOLO_HTTP_HPP
