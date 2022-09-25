#include <fstream>
#include <set>
#include <cassert>
#include <cstring>
#include "cfg.hpp"

namespace yolo
{
	void log(const std::string_view& message);

	namespace cfg
	{

		bool cfg::save(const std::filesystem::path& dest_cfg_filepath) const
		{
			std::ofstream file;
			file.open (dest_cfg_filepath);
			if(!file.is_open())
			{
				log("Failed to open '" + dest_cfg_filepath.string() + "' in 'cfg::save'");
				return false;
			}

			bool is_first_line = true;
			for(const auto& v : lines)
			{
				file << (is_first_line ? "" : "\n") << v;
				is_first_line = false;
			}
			file.close();
			return true;
		}

		static const std::string* get_line(const cfg& self, const std::string_view& section, const std::string_view& name)
		{
			const std::string section_str = "[" + std::string(section) + "]";
			bool is_in_section = section[0] == '\0';
			for(const auto& v : self.lines)
			{
				if(v.starts_with('['))
				{
					if(v.starts_with(section_str))
					{
						is_in_section = true;
					}
					else if(section[0] != '\0')
					{
						is_in_section = false;
					}
				}
				else if(is_in_section)
				{
					if(v.starts_with(name))
					{
						std::string remainder = v.substr(name.size());
						if(remainder.starts_with(' ') || remainder.starts_with('\t') || remainder.starts_with('='))
						{
							// bingo
							return &v;
						}
					}
				}
			}
			return nullptr;
		}


		bool cfg::get_value(const std::string_view& section, const std::string_view& name, std::string& target) const
		{
			if(const std::string* pLine = get_line(*this, section, name))
			{
				std::string line = *pLine;
				while(!line.empty() && (line[0] != '='))
				{
					line.erase(0, 1);
				}
				if(!line.empty() && (line[0] == '='))
				{
					line.erase(0, 1);
				}
				while(!line.empty() && (line[0] == ' ' || line[0] == '\t'))
				{
					line.erase(0, 1);
				}
				target = std::move(line);
				return true;
			}
			return false;
		}

		bool cfg::get_value(const std::string_view& section, const std::string_view& name, uint32_t& target) const
		{
			std::string value;
			if(!get_value(section, name, value))
			{
				return true;
			}
			char *endptr;
			uint32_t v = (int)strtol(value.c_str(), &endptr, 10);
			if (*endptr != '\0')
			{
				return false;
			}
			target = v;
			return true;
		}

		static void parse_line(std::string& line, std::optional<std::string>& scope, const load_args& load_args)
		{
			while(!line.empty() && (line[0] == ' ' || line[0] == '\t'))
			{
				line.erase(0, 1);
			}
			bool meets_scope = !scope.has_value();
			for(const auto& v : load_args.predefinitions)
			{
				if(scope.has_value())
				{
					if(scope->starts_with("NOT_"))
					{
						std::string not_key = &scope->c_str()[4];
						meets_scope |= not_key != v;
					}
					else
					{
						meets_scope |= *scope == v;
					}
				}

				if(line.find("@ifdef ") != std::string::npos)
				{
					std::string key = &line.c_str()[strlen("@ifdef ")];
					while(!key.empty() && *key.begin() == ' ')
					{
						key.erase(0, 1);
					}
					while(!key.empty() && *key.rbegin() == ' ')
					{
						key.pop_back();
					}

					scope = key;
					line.insert(line.begin(), '#');
				}
				else if(line.find("@else") != std::string::npos)
				{
					scope = "NOT_" + scope.value_or("");
					line.insert(line.begin(), '#');
				}
				else if(line.find("@endif") != std::string::npos)
				{
					scope = std::nullopt;
					line.insert(line.begin(), '#');
				}
			}
			if(!meets_scope)
			{
				line.insert(line.begin(), '#');
			}
			else
			{
				while(line.find("${") != std::string::npos)
				{
					auto start = line.find("${");
					auto end = line.find('}');
					std::string key = line.substr((start+2), end - (start+2));
					auto v = load_args.variables.find(key);
					std::string replacement;
					if(v != load_args.variables.end())
					{
						replacement = v->second;
					}
					else
					{
						log("Error parsing line '" + line + "'. Unknown variable name");
						replacement = "_PARSING_ERROR_";
					}
					line.erase(start, (end+1) - start);
					line.insert(start, replacement);
				}
			}
		}

		std::optional<cfg> load(const std::string_view& textfile_content, const load_args& load_args)
		{
			std::optional<std::string> scope;
			cfg ret;
			std::string line;
			for(const auto& v : textfile_content)
			{
				if(v != '\n')
				{
					line += v;
				}
				else
				{
					parse_line(line, scope, load_args);
					ret.lines.push_back(line);
					line.clear();
				}
			}
			if(!line.empty())
			{
				ret.lines.push_back(line);
			}
			return ret;
		}
	}
}