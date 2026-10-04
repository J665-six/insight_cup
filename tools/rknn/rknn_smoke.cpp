#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace {

void validate_model(const char* path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error(std::string("cannot open model: ") + path);
  }

  const std::streamsize size = stream.tellg();
  if (size <= 0) {
    throw std::runtime_error(std::string("model is empty: ") + path);
  }
}

void check(int result, const char* operation) {
  if (result != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(result));
  }
}

std::string dimensions(const rknn_tensor_attr& attr) {
  std::string value = "[";
  for (uint32_t index = 0; index < attr.n_dims; ++index) {
    if (index != 0) {
      value += ",";
    }
    value += std::to_string(attr.dims[index]);
  }
  return value + "]";
}

void print_tensor(const char* kind, const rknn_tensor_attr& attr) {
  std::cout << kind << "[" << attr.index << "] name=" << attr.name
            << " shape=" << dimensions(attr)
            << " format=" << get_format_string(attr.fmt)
            << " type=" << get_type_string(attr.type)
            << " elements=" << attr.n_elems << " bytes=" << attr.size
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL.rknn\n";
    return 2;
  }

  rknn_context context = 0;
  std::vector<rknn_output> outputs;
  bool outputs_acquired = false;

  try {
    validate_model(argv[1]);
    check(rknn_init(&context, argv[1], 0, RKNN_FLAG_COLLECT_PERF_MASK, nullptr),
          "rknn_init");

    rknn_sdk_version version{};
    check(rknn_query(context, RKNN_QUERY_SDK_VERSION, &version,
                     sizeof(version)),
          "RKNN_QUERY_SDK_VERSION");
    std::cout << "API: " << version.api_version << '\n'
              << "driver: " << version.drv_version << '\n';

    check(rknn_set_core_mask(context, RKNN_NPU_CORE_0),
          "rknn_set_core_mask(RKNN_NPU_CORE_0)");

    rknn_input_output_num count{};
    check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)),
          "RKNN_QUERY_IN_OUT_NUM");
    std::cout << "inputs=" << count.n_input << " outputs=" << count.n_output
              << '\n';
    if (count.n_input != 1) {
      throw std::runtime_error("smoke test currently expects one model input");
    }

    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr,
                     sizeof(input_attr)),
          "RKNN_QUERY_INPUT_ATTR");
    print_tensor("input", input_attr);

    std::vector<rknn_tensor_attr> output_attrs(count.n_output);
    for (uint32_t index = 0; index < count.n_output; ++index) {
      output_attrs[index].index = index;
      check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[index],
                       sizeof(output_attrs[index])),
            "RKNN_QUERY_OUTPUT_ATTR");
      print_tensor("output", output_attrs[index]);
    }

    std::vector<float> input_data(input_attr.n_elems, 0.0F);
    rknn_input input{};
    input.index = 0;
    input.buf = input_data.data();
    input.size = static_cast<uint32_t>(input_data.size() * sizeof(float));
    input.pass_through = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = input_attr.fmt;
    check(rknn_inputs_set(context, 1, &input), "rknn_inputs_set");

    outputs.resize(count.n_output);
    for (uint32_t index = 0; index < count.n_output; ++index) {
      outputs[index].index = index;
      outputs[index].want_float = 1;
      outputs[index].is_prealloc = 0;
    }

    const auto started = std::chrono::steady_clock::now();
    check(rknn_run(context, nullptr), "rknn_run");
    check(rknn_outputs_get(context, count.n_output, outputs.data(), nullptr),
          "rknn_outputs_get");
    outputs_acquired = true;
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started);

    rknn_perf_run performance{};
    check(rknn_query(context, RKNN_QUERY_PERF_RUN, &performance,
                     sizeof(performance)),
          "RKNN_QUERY_PERF_RUN");
    std::cout << std::fixed << std::setprecision(3)
              << "first inference wall time: " << elapsed.count() << " ms\n"
              << "NPU run duration: "
              << static_cast<double>(performance.run_duration) / 1000.0
              << " ms\n";

    check(rknn_outputs_release(context, count.n_output, outputs.data()),
          "rknn_outputs_release");
    outputs_acquired = false;
    check(rknn_destroy(context), "rknn_destroy");
    context = 0;
    std::cout << "NPU smoke test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    if (outputs_acquired && context != 0) {
      rknn_outputs_release(context, static_cast<uint32_t>(outputs.size()),
                           outputs.data());
    }
    if (context != 0) {
      rknn_destroy(context);
    }
    return 1;
  }
}
