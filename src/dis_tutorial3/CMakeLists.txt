cmake_minimum_required(VERSION 3.5)
project(dis_tutorial3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)

install(DIRECTORY
  launch
  config
  worlds
  maps
  DESTINATION share/${PROJECT_NAME}/
)

install(DIRECTORY scripts/
        DESTINATION share/${PROJECT_NAME}/scripts
        FILES_MATCHING PATTERN "*.txt")

install(PROGRAMS
  scripts/detect_people.py
  scripts/robot_commander.py
  scripts/detect_rings.py
  scripts/detect_people1.py
  scripts/speak.py
  scripts/rings_mask.py
  # scripts/voronoi_path.py
  scripts/skeletonized_path.py
  # scripts/kretanje.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
