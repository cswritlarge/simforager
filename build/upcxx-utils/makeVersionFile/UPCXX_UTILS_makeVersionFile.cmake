#
# This file is autogenerated by GetGitVersion.cmake
#
file(READ /carc/scratch/projects/melaniem2016345/simforager/build/upcxx-utils/makeVersionFile/UPCXX_UTILS_VERSION     UPCXX_UTILS_VERSION_RAW LIMIT 1024)
string(STRIP "${UPCXX_UTILS_VERSION_RAW}" UPCXX_UTILS_VERSION_AND_BRANCH)
separate_arguments(UPCXX_UTILS_VERSION_AND_BRANCH)
list(GET UPCXX_UTILS_VERSION_AND_BRANCH 0 UPCXX_UTILS_VERSION)
list(LENGTH UPCXX_UTILS_VERSION_AND_BRANCH UPCXX_UTILS_VERSION_AND_BRANCH_LEN)
if (UPCXX_UTILS_VERSION_AND_BRANCH_LEN GREATER 1)
  list(GET UPCXX_UTILS_VERSION_AND_BRANCH 1 UPCXX_UTILS_BRANCH)
else()
  set(UPCXX_UTILS_BRANCH)
endif()

set(UPCXX_UTILS_BUILD_DATE "20241002_143612")

set(UPCXX_UTILS_VERSION_STRING  "${UPCXX_UTILS_VERSION}")
message(STATUS "Building UPCXX_UTILS version ${UPCXX_UTILS_VERSION_STRING} on branch ${UPCXX_UTILS_BRANCH}")
configure_file(/carc/scratch/projects/melaniem2016345/simforager/build/upcxx-utils/makeVersionFile/version.cpp.in /carc/scratch/projects/melaniem2016345/simforager/build/upcxx-utils/makeVersionFile/___UPCXX_UTILS_AUTOGEN_version.c)

