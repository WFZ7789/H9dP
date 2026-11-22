#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace h9dp {

struct HimawariScan {
    // B08, B09, B10 and B16 HSD files, in any order.
    std::array<std::filesystem::path, 4> files;
    // All scans must either specify minutes on a common time axis or omit them
    // to use the mean of the four observation midpoints from the DAT headers.
    std::optional<double> minutes;
};

struct StormCenter {
    double latitude_deg;
    double longitude_deg;
};

class InferenceBest {
public:
    // Loads one or more exported V1 posteriors once. Multiple models are averaged.
    explicit InferenceBest(const std::filesystem::path& model);
    explicit InferenceBest(const std::vector<std::filesystem::path>& models);

    // Predict dP in hPa from one to six scans, sorted internally by scan time.
    // Throws std::exception on invalid inputs; concurrent const calls are safe.
    [[nodiscard]] double predict(const std::vector<HimawariScan>& scans, StormCenter center) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace h9dp
