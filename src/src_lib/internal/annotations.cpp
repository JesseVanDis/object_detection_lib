
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include "annotations.hpp"

namespace yolo
{
	void log(const std::string_view& message);

	static std::optional<std::filesystem::path> find_related_image_filepath(const std::filesystem::path& filepath_txt);

	namespace annotations
	{
		std::optional<annotation> annotation::load(const std::string_view& txt_line)
		{
			annotation v;
			char section[128];
			size_t start = 0;
			size_t category = 0;
			for(size_t i=0; i<=txt_line.size(); i++)
			{
				char c = i < txt_line.size() ? txt_line[i] : ' ';
				if(c == ' ')
				{
					memcpy(section, &txt_line[start], i - start);
					section[i - start] = '\0';
					start = (i+1);
					if(category == 0)
					{
						v.class_id = atoi(section);
					}
					else if(category == 1)
					{
						v.x = atof(section);
					}
					else if(category == 2)
					{
						v.y = atof(section);
					}
					else if(category == 3)
					{
						v.w = atof(section);
					}
					else if(category == 4)
					{
						v.h = atof(section);
					}
					category++;
				}
			}

			if(v.class_id != ~0u &&
			   v.x >= 0.0f && v.x <= 1.0f &&
			   v.y >= 0.0f && v.y <= 1.0f &&
			   v.w >= 0.0f && v.w <= 1.0f &&
			   v.h >= 0.0f && v.h <= 1.0f)
			{
				return v;
			}
			return std::nullopt;

			/*
			 0 0.1044921875 0.93212890625 0.08203125 0.0478515625
			0 0.18017578125 0.93017578125 0.0830078125 0.0498046875
			0 0.255859375 0.9287109375 0.083984375 0.052734375
			0 0.3447265625 0.92529296875 0.080078125 0.0458984375
			0 0.42724609375 0.9228515625 0.0810546875 0.048828125
			0 0.509765625 0.92041015625 0.08203125 0.0498046875
			0 0.61083984375 0.91845703125 0.0810546875 0.0478515625
			 */
		}

		std::optional<annotations> annotations::load(const std::filesystem::path& filepath_txt)
		{
			if(!std::filesystem::exists(filepath_txt))
			{
				log("Failed to find '" + filepath_txt.string() + "'");
				return std::nullopt;
			}

			auto related_image_path = find_related_image_filepath(filepath_txt);
			if(!related_image_path.has_value())
			{
				log("Failed to find the image related to '" + filepath_txt.string() + "'. ( image must be txt filename, with png or jpg extension, and should be in the same folder )");
				return std::nullopt;
			}

			annotations v;
			v.filename_txt = std::filesystem::weakly_canonical(std::filesystem::absolute(filepath_txt));
			v.filename_img = std::filesystem::weakly_canonical(std::filesystem::absolute(*related_image_path));

			std::string line;
			std::ifstream file(filepath_txt);
			if(!file.is_open())
			{
				log("Failed to open '" + filepath_txt.string() + "'");
			}

			bool ok = true;
			while(getline(file, line))
			{
				if(auto annotation = annotation::load(line))
				{
					v.data.push_back(*annotation);
				}
				else
				{
					log("Failed to parse line '" + line + "' from file '" + filepath_txt.string() + "'");
					ok = false;
					break;
				}
			}
			file.close();

			if(!ok)
			{
				return std::nullopt;
			}
			return v;
		}

		std::optional<annotations_collection> annotations_collection::load(const std::filesystem::path& folder_path)
		{
			if(!std::filesystem::exists(folder_path))
			{
				log("Failed to find '" + folder_path.string() + "'");
				return std::nullopt;
			}

			annotations_collection collection;

			for(const auto& v : std::filesystem::directory_iterator(folder_path))
			{
				if(v.path().extension() == ".txt")
				{
					auto annotations = annotations::load(v.path());
					if(!annotations.has_value())
					{
						log("Failed to load '" + v.path().string() + "'. Skipping this file");
					}
					else
					{
						collection.data.push_back(std::move(*annotations));
					}
				}
			}

			return collection;
		}

		uint32_t annotations_collection::num_classes() const
		{
			int highest_class_index = -1;
			for(const auto& a : *this)
			{
				auto i = std::max_element(a.begin(), a.end(), [](const annotation& b1, const annotation& b2){return b1.class_id < b2.class_id;});
				if(i != a.end())
				{
					highest_class_index = std::max(highest_class_index, (int)i->class_id);
				}
			}
			return (uint32_t)(highest_class_index+1);
		}

		bool annotations_collection::save_darknet_txt(const std::filesystem::path& dest_filepath) const
		{
			std::filesystem::path folder = dest_filepath;
			folder.remove_filename();
			if(!std::filesystem::exists(folder))
			{
				std::filesystem::create_directories(folder);
			}

			std::ofstream file;
			file.open (dest_filepath);
			if(!file.is_open())
			{
				return false;
			}

			bool is_first_line = true;
			for(const auto& v : *this)
			{
				const auto absolute_path = std::filesystem::weakly_canonical(std::filesystem::absolute(v.filename_img));
				file << (is_first_line ? "" : "\n") << absolute_path.string(); // yes, only the 'filename_img'. darknet should automatically find the relevant txt file by changing the extension
				is_first_line = false;
			}
			file.close();
			return true;
		}

		bool annotations_collection::save_darknet_names(const std::filesystem::path& dest_names_filepath, const std::unordered_map<uint32_t, std::string>& names_map) const
		{
			uint32_t num_of_classes = num_classes();
			if(num_of_classes == 0)
			{
				return false;
			}

			std::filesystem::path folder = dest_names_filepath;
			folder.remove_filename();
			if(!std::filesystem::exists(folder))
			{
				std::filesystem::create_directories(folder);
			}

			std::ofstream file;
			file.open (dest_names_filepath);
			if(!file.is_open())
			{
				return false;
			}

			bool is_first_line = true;
			for(int i=0; i<(int)num_of_classes; i++)
			{
				std::string name = std::to_string(i);
				auto match = names_map.find((uint32_t)i);
				if(match != names_map.end())
				{
					name = match->second;
				}
				else if(num_of_classes == 1)
				{
					name = ".";
				}
				file << (is_first_line ? "" : "\n") << name;
				is_first_line = false;
			}
			file.close();
			return true;
		}

		void annotations_collection::split_to_training_and_valid_collections(annotations_collection& dest_training, annotations_collection& dest_eval, float ratio) const
		{
			srand(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

			dest_training.clear();
			dest_eval.clear();

			for(const auto& v : *this)
			{
				const bool should_eval = ((float)rand() / (float)RAND_MAX) <= ratio; // NOLINT
				if(should_eval)
				{
					dest_eval.data.push_back(v);
				}
				else
				{
					dest_training.data.push_back(v);
				}
			}

			if(dest_eval.data.empty())
			{
				log("WARNING: We have no evaluation data. Make sure the validation_ratio is high enough");
			}
		}
	}

	static std::string remove_extension(const std::string& filename)
	{
		size_t lastdot = filename.find_last_of('.');
		if (lastdot == std::string::npos) return filename;
		return filename.substr(0, lastdot);
	}

	static std::optional<std::filesystem::path> find_related_image_filepath(const std::filesystem::path& filepath_txt)
	{
		const std::string path_without_extension = remove_extension(filepath_txt);
		if(std::filesystem::exists(path_without_extension + ".png"))
		{
			return path_without_extension + ".png";
		}
		if(std::filesystem::exists(path_without_extension + ".jpg"))
		{
			return path_without_extension + ".jpg";
		}
		return std::nullopt;
	}

}

