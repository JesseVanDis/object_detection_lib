#include <yolo.hpp>
#include "server.hpp"


namespace yolo::server
{
	server::server(std::unique_ptr<server_internal>&& v)
		: m_internal(std::move(v))
	{}

	server::~server() = default;

	server_internal::server_internal(init_args&& init_args)
		: m_init_args(std::move(init_args))
	{
	}

	bool server_internal::is_running() const
	{
		return false;
	}
}
