cmake_minimum_required(VERSION 3.16)
project(fr5_bringup)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_vla_pick_target
    test/test_vla_pick_target.py
  )
endif()

install(DIRECTORY urdf config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/a1_move_and_read.py
  scripts/a2_gripper.py
  scripts/a2_tcp_calibrate.py
  scripts/b1_capture_points.py
  scripts/b2_fit_transform.py
  scripts/b3_pick_point.py
  scripts/b3_hover.py
  scripts/d0_point_grab.py
  scripts/vla_pick_target.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
