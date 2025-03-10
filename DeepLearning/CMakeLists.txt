cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Master_Thesis)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Set the executable name and main source file according to your setup
add_executable(DeepLearning DeepLearning.cpp)
target_link_libraries(DeepLearning "${TORCH_LIBRARIES}")
set_property(TARGET DeepLearning PROPERTY CXX_STANDARD 17)

# Optional Windows-specific code block for handling DLLs (you can ignore this on Kubuntu)
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET DeepLearning
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:DeepLearning>)
endif (MSVC)


