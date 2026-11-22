#pragma once

#include "inference_best.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace h9dp::v1 {

inline constexpr int frames = 6;
inline constexpr int crop_size = 500;
inline constexpr std::array<int, 6> dimensions{320, 400, 320, 288, 60, 672};
enum Group { eye, wall, relief, waves, motion, sector };

struct Crop {
    std::array<std::vector<std::int16_t>, 4> temperature;
    std::vector<std::uint8_t> valid;
    double x, y;
    double minutes;
};

struct Features {
    std::array<double, frames> minutes;
    // Flat frame-major groups; NaN denotes a missing value.
    std::array<std::vector<float>, dimensions.size()> groups;
};

Crop load_crop(const HimawariScan& scan, StormCenter center);
Features describe(std::vector<Crop> crops);

// Internal regression path: evaluate packed, cached Python descriptors.
void predict_packed(const std::filesystem::path& input,
                    const std::vector<std::filesystem::path>& models, std::ostream& out);

} // namespace h9dp::v1
