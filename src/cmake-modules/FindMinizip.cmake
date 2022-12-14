FIND_PACKAGE(ZLIB)

IF(MINIZIP_DIR)
  FIND_PATH(MINIZIP_INCLUDE_DIR_zh NAMES minizip/zip.h NO_DEFAULT_PATH PATHS ${MINIZIP_DIR}/include ${MINIZIP_DIR})
  FIND_PATH(MINIZIP_INCLUDE_DIR_unzh NAMES minizip/unzip.h NO_DEFAULT_PATH PATHS ${MINIZIP_DIR}/include ${MINIZIP_DIR})
  FIND_LIBRARY(MINIZIP_LIBRARY NAMES minizip NO_DEFAULT_PATH PATHS ${MINIZIP_DIR}/lib ${MINIZIP_DIR})
ELSE()
  FIND_PATH(MINIZIP_INCLUDE_DIR_zh NAMES minizip/zip.h PATHS /include /usr/include /usr/local/include /opt/local/include)
  FIND_PATH(MINIZIP_INCLUDE_DIR_unzh NAMES minizip/unzip.h PATHS /include /usr/include /usr/local/include /opt/local/include)
  FIND_LIBRARY(MINIZIP_LIBRARY NAMES minizip)
ENDIF()

IF (MINIZIP_INCLUDE_DIR_unzh AND MINIZIP_INCLUDE_DIR_zh AND MINIZIP_LIBRARY)
  SET(MINIZIP_INCLUDE_DIRS "${MINIZIP_INCLUDE_DIR_unzh};${MINIZIP_INCLUDE_DIR_zh};${ZLIB_INCLUDE_DIRS}")
  SET(MINIZIP_LIBRARIES ${MINIZIP_LIBRARY} ${ZLIB_LIBRARIES})
  SET(MINIZIP_FOUND true)
ENDIF()
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Minizip DEFAULT_MSG MINIZIP_LIBRARIES MINIZIP_INCLUDE_DIRS)