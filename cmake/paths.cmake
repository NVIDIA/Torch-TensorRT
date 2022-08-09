include(GNUInstallDirs)

#Set output directory names for libraries and binaries
set(LIBRARY_OUTPUT_DIRECTORY "lib")
set(ARCHIVE_OUTPUT_DIRECTORY "lib")
set(RUNTIME_OUTPUT_DIRECTORY "bin")
set(HEADERS_OUTPUT_DIRECTORY "include")

#Set target ouput directory in the build directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${ARCHIVE_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${LIBRARY_OUTPUT_DIRECTORY}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${RUNTIME_OUTPUT_DIRECTORY}")
