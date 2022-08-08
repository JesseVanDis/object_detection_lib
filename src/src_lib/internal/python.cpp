#include <Python.h>
#include <sstream>
#include <unordered_map>
#include "python.hpp"

namespace yolo
{
	void log(const std::string_view& message);
}

namespace yolo::python
{
	size_t s_num_active_sessions = 0;

	std::unordered_map<PyObject*, instance*> s_ptr_to_instance;

	static instance* instance_at(PyObject* ptr)
	{
		auto it = s_ptr_to_instance.find(ptr);
		if(it != s_ptr_to_instance.end())
		{
			return it->second;
		}
		return nullptr;
	}

	static PyObject* app_print(PyObject* self, PyObject* args)
	{
		const char* text_cstr;
		unsigned long ret = ~0u;
		if(PyArg_ParseTuple(args, "s", &text_cstr))
		{
			std::string text = text_cstr;
#ifndef SLASH_R_SUPPORTED
			text.erase(std::remove_if(text.begin(), text.end(), [](char v){return v == '\r';}), text.end());
			const bool only_contains_spaces = std::find_if(text.begin(), text.end(), [](char v){return v != ' ';}) == text.end();
			if(only_contains_spaces)
			{
				text.clear();
			}
#endif

			if(!text.empty() && *text.rbegin() == '\n')
			{
				text.pop_back();
			}
			if(text.empty())
			{
				ret = strlen(text_cstr);
			}
			else
			{
				if(instance* p_instance = instance_at(self))
				{
					if(p_instance->args.print_callback.has_value())
					{
						(*p_instance->args.print_callback)("py: " + text);
						ret = text.size();
					}
				}
			}
			if(ret == ~0u)
			{
				log("py: " + text);
				ret = text.size();
			}
		}
		else
		{
			log("Error: Failed to parse tuple in 'app.print'");
		}
		if(ret == ~0u)
		{
			ret = 0;
		}
		return PyLong_FromLong((long)ret);
	}

	static PyMethodDef s_app_methods[] = {
			{"print",  app_print, METH_VARARGS, "print text to application."},
			{nullptr, nullptr, 0, nullptr}        // Sentinel
	};

	static struct PyModuleDef s_custom_module = {
			.m_base = PyModuleDef_HEAD_INIT,
			.m_name = "app",    // name of module
			.m_doc = nullptr, 	// module documentation, may be NULL
			.m_size = -1,       // size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
			.m_methods = s_app_methods,
			.m_slots = nullptr,
			.m_traverse = nullptr,
			.m_clear = nullptr,
			.m_free = nullptr
	};

	PyMODINIT_FUNC PyInit_app(void)
	{
		return PyModule_Create(&s_custom_module);
	}

	std::unique_ptr<instance> new_instance(const init_args& init_args)
	{
		std::unique_ptr<instance> ptr = std::unique_ptr<instance>(new instance(init_args));
		if(!ptr->initialized())
		{
			return nullptr;
		}
		return ptr;
	}

	instance::instance(const init_args& init_args)
		: args(init_args)
	{
		if(s_num_active_sessions > 0)
		{
			log("Error: only 1 python session allowed at a time");
			m_initialized = false;
			return;
		}

		// Add built-in module
		if (PyImport_AppendInittab("app", PyInit_app) == -1) {
			log("Error: could not extend in-built modules table");
			m_initialized = false;
			return;
		}

		Py_Initialize();

		PyObject* p_module = PyImport_ImportModule("app");
		if (!p_module)
		{
			log("Error: could not import module 'app'");
			m_initialized = false;
			return;
		}

		s_ptr_to_instance.insert({p_module, this});

		m_initialized = true;

		//run("import app");

		if(args.print_callback.has_value())
		{
			std::stringstream python_command;
			auto py = [&python_command]() -> std::stringstream&
			{
				python_command << std::endl;
				return python_command;
			};

			py() 	<< "import _io";
			py() 	<< "import sys";
			py() 	<< "import app";
			py() 	<< "import os";
			py() 	<< "";
			py() 	<< "def custom_get_terminal_size():";
			py() 	<< "	return (80, 24)";
			py() 	<< "";
			py() 	<< "os.get_terminal_size = custom_get_terminal_size";
			py() 	<< "";
			py() 	<< "class CustomPrint(_io.TextIOWrapper):";
			py() 	<< "	def __init__(self, *args, **kwargs):";
			py() 	<< "		super().__init__(*args, **kwargs)";
			py() 	<< "	";
			py() 	<< "	def write(self, text):";
			//py() 	<< "		super().write(text)";
			py() 	<< "		app.print(text)";
			py() 	<< "";
			py() 	<< "sys.stdout = CustomPrint(sys.stdout.buffer)";
			py() 	<< "";

			run(python_command.str());
		}
	}

	instance::~instance()
	{
		Py_Finalize();
		s_num_active_sessions--;
		while(!s_ptr_to_instance.empty())
		{
			auto it = std::find_if(s_ptr_to_instance.begin(), s_ptr_to_instance.end(), [this](const auto& v){ return v.second == this; });
			if(it != s_ptr_to_instance.end())
			{
				s_ptr_to_instance.erase(it);
			}
			else
			{
				break;
			}
		}
	}

	bool instance::initialized() const
	{
		return m_initialized;
	}

	void instance::run(const std::string_view& python_code)
	{
		std::stringstream ss;
		ss << std::endl;
		ss << python_code;
		ss << std::endl;
		std::string code = ss.str();
		const char* code_cstr = code.c_str();
		PyRun_SimpleString(code_cstr);
	}

	instance& instance::operator << (const std::string_view& python_line)
	{
		run(python_line);
		return *this;
	}

	instance& instance::operator << (const std::string& python_line)
	{
		run(python_line);
		return *this;
	}

	instance& instance::operator << (const char* python_line)
	{
		run(python_line);
		return *this;
	}

	std::stringstream& builder::operator<<(const std::string_view& python_line)
	{
		m_code << std::endl << python_line;
		return m_code;
	}

	std::stringstream& builder::operator<<(const std::string& python_line)
	{
		m_code << std::endl << python_line;
		return m_code;
	}

	std::stringstream& builder::operator<<(const char* python_line)
	{
		m_code << std::endl << python_line;
		return m_code;
	}

	void builder::run()
	{
		std::string code = m_code.str();
		if(!code.empty())
		{
			const char* code_cstr = code.c_str();
			m_instance.run(code_cstr);
		}
		m_code = {};
	}
}
