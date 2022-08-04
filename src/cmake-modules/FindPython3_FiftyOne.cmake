
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_MODULE_NAME fiftyone)

execute_process (
        COMMAND "${PYTHON_EXECUTABLE}" -E -c "import pkgutil; print(1 if pkgutil.find_loader(\"${PYTHON_MODULE_NAME}\") else 0)"
        RESULT_VARIABLE STATUS
        OUTPUT_VARIABLE P
        ERROR_VARIABLE  ERROR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT ERROR MATCHES "'import site' failed|ImportError: No module named site")
    if (P EQUAL 1)
        set(Python3_FiftyOne_FOUND TRUE)
    endif ()
endif ()

