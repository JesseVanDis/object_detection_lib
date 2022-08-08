#ifndef ALL_YOLO_PYTHON_HPP
#define ALL_YOLO_PYTHON_HPP

#ifdef PYTHON3_FOUND

#include <memory>
#include <string_view>
#include <functional>

namespace yolo::python
{
	class instance;

	struct init_args
	{
		std::optional<std::function<void(const std::string_view& str)>> print_callback;
	};

	class builder
	{
			friend class instance;
		public:
			builder(builder&& other) noexcept : m_code(std::move(other.m_code)), m_instance(other.m_instance){}
			~builder() { run(); }

			std::stringstream& operator << (const std::string_view& python_line);
			std::stringstream& operator << (const std::string& python_line);
			std::stringstream& operator << (const char* python_line);

			void run();

		protected:
			explicit builder(instance& instance) : m_instance(instance){}

		private:
			builder(const builder& other) : m_instance(other.m_instance){}

			std::stringstream m_code;
			instance& m_instance;
	};

	class instance
	{
		protected:
			friend std::unique_ptr<instance> new_instance(const init_args& init_args);
			explicit instance(const init_args& init_args);

			[[nodiscard]] bool initialized() const;

		public:
			~instance();

			void run(const std::string_view& python_code);

			builder code_builder() { return yolo::python::builder(*this); }

			instance& operator << (const std::string_view& python_line);
			instance& operator << (const std::string& python_line);
			instance& operator << (const char* python_line);

			template<typename T>
			instance& operator << (const T& v) { return (*this) << std::to_string(v); }

			const init_args args;

		private:
			bool m_initialized = false;
	};

	std::unique_ptr<instance> new_instance(const init_args& init_args = {});
}

#endif

#endif //ALL_YOLO_PYTHON_HPP
