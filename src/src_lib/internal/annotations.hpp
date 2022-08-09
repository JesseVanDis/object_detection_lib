#ifndef ALL_YOLO_ANNOTATIONS_HPP
#define ALL_YOLO_ANNOTATIONS_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <optional>
#include <filesystem>
#include <unordered_map>

namespace yolo::annotations
{
	struct annotation
	{
		uint32_t class_id = ~0u;

		/// starting position X of the annotation. range 0 - 1
		float x = -1;

		/// starting position Y of the annotation. range 0 - 1
		float y = -1;

		/// width of the annotation. range 0 - 1
		float w = -1;

		/// height of the annotation. range 0 - 1
		float h = -1;

		static std::optional<annotation> load(const std::string_view& txt_line);
	};


	struct annotations
	{
		std::vector<annotation> data;
		std::filesystem::path	filename_txt;
		std::filesystem::path	filename_img;

		[[nodiscard]] auto 		begin() const 	{ return data.begin(); }
		[[nodiscard]] auto 		end() const 	{ return data.end(); }
		[[nodiscard]] auto 		empty() const 	{ return data.empty(); }
		[[nodiscard]] auto 		size() const 	{ return data.size(); }

		static std::optional<annotations> load(const std::filesystem::path& filepath_txt);
	};

	struct annotations_collection
	{
		std::vector<annotations> data;

		[[nodiscard]] auto 		begin() const 	{ return data.begin(); }
		[[nodiscard]] auto 		end() const 	{ return data.end(); }
		[[nodiscard]] auto 		empty() const 	{ return data.empty(); }
		[[nodiscard]] auto 		size() const 	{ return data.size(); }
		void 					clear() 		{ return data.clear(); }

								/// \param dest_txt_filepath the target filepath. example: ./train.txt
								/// \return true if the writing of the file succeeded ( assume true if you have enough space )
		bool 					save_darknet_txt(const std::filesystem::path& dest_txt_filepath) const; // NOLINT

								/// \param dest_names_filepath the target filepath. example: ./yolo.names
								/// \return true if the writing of the file succeeded ( assume true if you have enough space )
		bool 					save_darknet_names(const std::filesystem::path& dest_names_filepath, const std::unordered_map<uint32_t, std::string>& names_map = {}) const; // NOLINT

		[[nodiscard]] uint32_t	num_classes() const;

								/// \param dest_training target for training
								/// \param dest_eval  target for evaluation
								/// \param ratio  split ratio. range 0 - 1.  the smaller the value, the less evaluations
		void 					split_to_training_and_valid_collections(annotations_collection& dest_training, annotations_collection& dest_eval, float ratio) const;

		/// \param server_or_folder_path example: "/home/me/data"
		static std::optional<annotations_collection> load(const std::filesystem::path& folder_path);
	};

}

#endif //ALL_YOLO_ANNOTATIONS_HPP
