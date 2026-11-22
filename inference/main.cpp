#include "v1_detail.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int usage(const char* name) {
    std::cerr << "Usage: " << name << " LAT LON MODEL.v1m [MODEL.v1m ...] -- DAT DAT DAT DAT [...]\n"
              << "       " << name << " --features REQUEST.v1q MODEL.v1m [MODEL.v1m ...]\n";
    return 2;
}

double number(const char* value) {
    std::size_t consumed = 0;
    const std::string text(value);
    const double out = std::stod(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("Invalid latitude or longitude: " + text);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 4 && std::string_view(argv[1]) == "--features") {
            h9dp::v1::predict_packed(argv[2], {argv + 3, argv + argc}, std::cout);
            return 0;
        }
        if (argc < 9) {
            return usage(argv[0]);
        }
        const auto split = std::find_if(argv + 3, argv + argc, [](const char* arg) {
            return std::string_view(arg) == "--";
        });
        if (split == argv + 3 || split == argv + argc) {
            return usage(argv[0]);
        }
        const auto count = argv + argc - split - 1;
        if (count < 4 || count > 24 || count % 4 != 0) {
            return usage(argv[0]);
        }
        std::vector<h9dp::HimawariScan> scans;
        for (auto file = split + 1; file != argv + argc; file += 4) {
            scans.push_back({{file[0], file[1], file[2], file[3]}, std::nullopt});
        }
        const h9dp::StormCenter center{number(argv[1]), number(argv[2])};
        const h9dp::InferenceBest model(std::vector<std::filesystem::path>(argv + 3, split));
        std::cout << std::setprecision(17) << model.predict(scans, center) << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "inference_best: " << e.what() << '\n';
        return 1;
    }
}
