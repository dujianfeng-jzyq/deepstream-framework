# example-service：Python 服务单元，源码安装到载荷 services/example-service/。
dsf_add_python_service(example-service FILES main.py REQUIREMENTS requirements.txt)
dsf_add_python_test(example_service_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_example_service.py")
