# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code

# Include any dependencies generated for this target.
include CMakeFiles/evaluation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/evaluation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/evaluation.dir/flags.make

CMakeFiles/evaluation.dir/evaluation.cpp.o: CMakeFiles/evaluation.dir/flags.make
CMakeFiles/evaluation.dir/evaluation.cpp.o: evaluation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/evaluation.dir/evaluation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/evaluation.dir/evaluation.cpp.o -c /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/evaluation.cpp

CMakeFiles/evaluation.dir/evaluation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evaluation.dir/evaluation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/evaluation.cpp > CMakeFiles/evaluation.dir/evaluation.cpp.i

CMakeFiles/evaluation.dir/evaluation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evaluation.dir/evaluation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/evaluation.cpp -o CMakeFiles/evaluation.dir/evaluation.cpp.s

# Object files for target evaluation
evaluation_OBJECTS = \
"CMakeFiles/evaluation.dir/evaluation.cpp.o"

# External object files for target evaluation
evaluation_EXTERNAL_OBJECTS =

evaluation: CMakeFiles/evaluation.dir/evaluation.cpp.o
evaluation: CMakeFiles/evaluation.dir/build.make
evaluation: /usr/lib/x86_64-linux-gnu/libmpfr.so
evaluation: /usr/lib/x86_64-linux-gnu/libgmp.so
evaluation: CMakeFiles/evaluation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable evaluation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/evaluation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/evaluation.dir/build: evaluation

.PHONY : CMakeFiles/evaluation.dir/build

CMakeFiles/evaluation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/evaluation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/evaluation.dir/clean

CMakeFiles/evaluation.dir/depend:
	cd /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code /home/pingping/Dropbox/UofSC/Research_Assistant/Point_Manifold_Upconvolution/code/MDPU1K/evaluation_code/CMakeFiles/evaluation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/evaluation.dir/depend

