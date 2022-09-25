
#ifndef ALL_YOLO_CFG_HPP
#define ALL_YOLO_CFG_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <optional>
#include <map>
#include <string_view>

namespace yolo::cfg
{
	struct load_args
	{
		std::vector<std::string> predefinitions;
		std::map<std::string, std::string> variables;
	};

	struct cfg
	{
		std::vector<std::string> lines;

		bool save(const std::filesystem::path& dest_cfg_filepath) const; // NOLINT
		bool get_value(const std::string_view& section, const std::string_view& name, std::string& target) const;
		bool get_value(const std::string_view& section, const std::string_view& name, uint32_t& target) const;
	};

	std::optional<cfg> load(const std::string_view& textfile_content, const load_args& load_args);
}


#endif //ALL_YOLO_CFG_HPP
