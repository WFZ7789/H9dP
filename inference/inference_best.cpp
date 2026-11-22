#include "v1_detail.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace h9dp::v1 {
namespace {

constexpr int stats = 3;
constexpr std::array<double, 3> sequence_widths{0.035, 0.14, 0.55};
constexpr std::array<double, 2> spline_widths{0.7, 1.8};
constexpr int knots = 9;
constexpr double nan = std::numeric_limits<double>::quiet_NaN();
constexpr double minus_inf = -std::numeric_limits<double>::infinity();

class Reader {
public:
    explicit Reader(const std::filesystem::path& path) : name_(path.string()), file_(path, std::ios::binary) {
        static_assert(std::endian::native == std::endian::little && sizeof(float) == 4 && sizeof(double) == 8
                      && std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
                      "The V1 binary format requires a little-endian IEEE-754 host");
        if (!file_) {
            fail("Cannot open file");
        }
    }

    void magic(std::string_view expected) {
        const auto data = array<char>(expected.size());
        if (std::string_view(data.data(), data.size()) != expected) {
            fail("Wrong V1 file type or version");
        }
    }

    template<class T> T scalar() {
        T value;
        bytes(&value, sizeof(T));
        return value;
    }

    template<class T> std::vector<T> array(std::size_t n) {
        std::vector<T> values(n);
        if (n != 0) {
            bytes(values.data(), n * sizeof(T));
        }
        return values;
    }

    template<class T> std::vector<T> finite(std::size_t n) {
        auto values = array<T>(n);
        if (!std::all_of(values.begin(), values.end(), [](T x) { return std::isfinite(x); })) {
            fail("Non-finite V1 model coefficient");
        }
        return values;
    }

    void end() {
        if (file_.peek() != std::istream::traits_type::eof() || file_.bad()) {
            fail("Trailing bytes or unreadable end of file");
        }
    }

    [[noreturn]] void fail(std::string_view reason) const {
        throw std::runtime_error(name_ + ": " + std::string(reason));
    }

private:
    void bytes(void* data, std::size_t n) {
        if (!file_.read(static_cast<char*>(data), static_cast<std::streamsize>(n))) {
            fail("Truncated V1 file");
        }
    }

    std::string name_;
    std::ifstream file_;
};

struct Time {
    std::array<double, frames> lag{}, recent{}, sequence{};

    explicit Time(const Features& features) {
        const int d = dimensions[eye];
        double latest = minus_inf;
        for (int i = 0; i < frames; ++i) {
            const auto first = features.groups[eye].begin() + i * d;
            const bool valid = std::any_of(first, first + d, [](float x) { return std::isfinite(x); });
            if (valid && !std::isfinite(features.minutes[i])) {
                throw std::invalid_argument("An eye-valid frame has no finite time");
            }
            lag[i] = valid ? features.minutes[i] : minus_inf;
            latest = std::max(latest, lag[i]);
        }
        if (!std::isfinite(latest)) {
            throw std::invalid_argument("V1 needs at least one timed scan with usable eye-region pixels");
        }
        double total = 0;
        for (int i = 0; i < frames; ++i) {
            lag[i] -= latest;
            recent[i] = std::exp(lag[i] / 45.0);
            total += sequence[i] = std::exp(lag[i] / 180.0);
        }
        for (double& weight : sequence) {
            weight = static_cast<float>(weight / total); // The reference sequence kernel uses FP32 weights.
        }
    }
};

struct Normalizer {
    std::vector<std::uint32_t> column;
    std::vector<double> mean, scale;

    void read(Reader& src, int dim) {
        const auto n = src.scalar<std::uint32_t>();
        if (n > static_cast<unsigned>(dim)) {
            src.fail("Invalid V1 normalized dimensions");
        }
        column = src.array<std::uint32_t>(n);
        mean = src.finite<double>(n);
        scale = src.finite<double>(n);
        if ((!column.empty() && (column.back() >= static_cast<unsigned>(dim)
                                 || std::adjacent_find(column.begin(), column.end(), std::greater_equal{}) != column.end()))
            || std::any_of(scale.begin(), scale.end(), [](double x) { return x <= 0; })) {
            src.fail("Invalid V1 normalization columns or scales");
        }
    }

    double apply(double x, std::size_t i) const {
        return std::isfinite(x) ? std::clamp((x - mean[i]) / scale[i], -5.0, 5.0) : 0;
    }
};

// Recent decayed mean, latest feature, and recent minus mean of scans >= 60 min old.
std::array<double, stats> summarize(const std::vector<float>& x, int dim, int col, const Time& time) {
    double nearest = minus_inf, latest = nan, mass = 0, recent_sum = 0, past_sum = 0;
    int past_n = 0;
    for (int i = 0; i < frames; ++i) {
        const double value = x[i * dim + col];
        if (!std::isfinite(value) || !std::isfinite(time.lag[i])) {
            continue;
        }
        if (time.lag[i] > nearest) {
            nearest = time.lag[i];
            latest = value;
        }
        mass += time.recent[i];
        recent_sum += value * time.recent[i];
        if (time.lag[i] <= -60.0) {
            ++past_n;
            past_sum += value;
        }
    }
    const double recent = mass > 0 ? recent_sum / mass : nan;
    return {recent, latest, past_n > 0 ? recent - past_sum / past_n : nan};
}

template<class T>
double distance(std::span<const T> a, std::span<const T> b) {
    double out = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta = static_cast<double>(a[i]) - b[i];
        out += delta * delta;
    }
    return out / a.size();
}

struct Stat {
    Normalizer norm;
    double matern_mass = 0;
    std::vector<double> support, linear;
    std::array<std::vector<double>, spline_widths.size()> spline;

    void read(Reader& src, int dim, std::size_t n, bool has_spline) {
        norm.read(src, dim);
        const auto size = norm.column.size();
        if (!size) {
            return;
        }
        matern_mass = src.scalar<double>();
        if (!std::isfinite(matern_mass) || matern_mass < 0) {
            src.fail("Invalid V1 Matérn mass");
        }
        linear = src.finite<double>(size);
        support = src.finite<double>(size * n);
        if (has_spline) {
            for (auto& phi : spline) {
                phi = src.finite<double>(size * knots);
            }
        }
    }

    double predict(const std::vector<float>& values, int dim, int stat, const Time& time,
                   const std::vector<double>& dual) const {
        const std::size_t size = norm.column.size();
        if (!size) {
            return 0;
        }
        std::vector<double> x(size);
        for (std::size_t i = 0; i < size; ++i) {
            x[i] = norm.apply(summarize(values, dim, norm.column[i], time)[stat], i);
        }
        double result = std::inner_product(x.begin(), x.end(), linear.begin(), 0.0);
        for (std::size_t k = 0; k < spline.size() && !spline[k].empty(); ++k) {
            for (std::size_t c = 0; c < size; ++c) {
                double squared_norm = 0, dot = 0;
                for (int j = 0; j < knots; ++j) {
                    const double delta = (x[c] - (-3 + 0.75 * j)) / spline_widths[k];
                    const double phi = std::exp(-0.5 * delta * delta);
                    squared_norm += phi * phi;
                    dot += phi * spline[k][c * knots + j];
                }
                result += dot / std::sqrt(squared_norm);
            }
        }
        double matern = 0;
        for (std::size_t j = 0; j < dual.size(); ++j) {
            const double r = std::sqrt(3 * distance<double>(x, {support.data() + j * size, size}) + 1e-12);
            matern += dual[j] * (1 + r) * std::exp(-r);
        }
        return result + matern_mass * matern;
    }
};

struct Sequence {
    Normalizer norm;
    std::vector<float> support;
    std::vector<double> mass, weights, scale;

    void read(Reader& src, int dim, std::size_t n) {
        norm.read(src, dim);
        if (norm.column.empty()) {
            src.fail("Missing V1 sequence columns");
        }
        mass = src.finite<double>(sequence_widths.size());
        weights = src.finite<double>(n * frames);
        support = src.finite<float>(n * frames * norm.column.size());
        scale = src.finite<double>(n * sequence_widths.size());
        if (std::any_of(mass.begin(), mass.end(), [](double x) { return x < 0; })
            || std::any_of(weights.begin(), weights.end(), [](double x) { return x < 0 || x > 1; })
            || std::any_of(scale.begin(), scale.end(), [](double x) { return x <= 0; })) {
            src.fail("Invalid V1 sequence model weights or scales");
        }
    }

    std::array<double, sequence_widths.size()> gram(std::span<const float> a, std::span<const float> b,
                                                    const double* w, const double* z) const {
        std::array<double, sequence_widths.size()> sum{};
        const auto size = norm.column.size();
        for (int i = 0; i < frames; ++i) {
            if (w[i] == 0) {
                continue;
            }
            for (int j = 0; j < frames; ++j) {
                const double weight = w[i] * z[j];
                if (weight > 0) {
                    const double d = distance<float>(a.subspan(i * size, size), b.subspan(j * size, size));
                    for (std::size_t k = 0; k < sum.size(); ++k) {
                        sum[k] += weight * std::exp(-d / (2 * sequence_widths[k]));
                    }
                }
            }
        }
        return sum;
    }

    double predict(const std::vector<float>& values, int dim, const Time& time,
                   const std::vector<double>& dual) const {
        const auto size = norm.column.size();
        std::vector<float> x(frames * size);
        for (int f = 0; f < frames; ++f) {
            for (std::size_t c = 0; c < size; ++c) {
                x[f * size + c] = static_cast<float>(norm.apply(values[f * dim + norm.column[c]], c));
            }
        }
        auto self = gram(x, x, time.sequence.data(), time.sequence.data());
        for (double& value : self) {
            value = 1 / std::sqrt(std::max(value, 1e-6));
        }
        double result = 0;
        for (std::size_t j = 0; j < dual.size(); ++j) {
            const auto k = gram(x, {support.data() + j * x.size(), x.size()}, time.sequence.data(),
                                weights.data() + j * frames);
            for (std::size_t i = 0; i < k.size(); ++i) {
                result += dual[j] * mass[i] * k[i] * self[i] * scale[j * k.size() + i];
            }
        }
        return result;
    }
};

struct Model {
    double intercept;
    std::vector<double> dual;
    std::array<std::array<Stat, stats>, dimensions.size()> summaries;
    std::array<std::optional<Sequence>, dimensions.size()> sequences;

    explicit Model(const std::filesystem::path& path) {
        Reader src(path);
        src.magic("H9V1MOD1");
        const auto n = src.scalar<std::uint32_t>();
        intercept = src.scalar<double>();
        if (n == 0 || n > 4096 || !std::isfinite(intercept)) {
            src.fail("Invalid V1 posterior header");
        }
        dual = src.finite<double>(n);
        for (std::size_t g = 0; g < dimensions.size(); ++g) {
            for (int s = 0; s < stats; ++s) {
                summaries[g][s].read(src, dimensions[g], n, g <= relief && s == 0);
            }
            if (g == waves || g == sector) {
                sequences[g].emplace().read(src, dimensions[g], n);
            }
        }
        src.end();
    }

    double predict(const Features& input, const Time& time) const {
        double result = intercept;
        for (std::size_t g = 0; g < dimensions.size(); ++g) {
            for (int s = 0; s < stats; ++s) {
                result += summaries[g][s].predict(input.groups[g], dimensions[g], s, time, dual);
            }
            if (sequences[g]) {
                result += sequences[g]->predict(input.groups[g], dimensions[g], time, dual);
            }
        }
        return result;
    }
};

std::vector<Model> load_models(const std::vector<std::filesystem::path>& paths) {
    if (paths.empty()) {
        throw std::invalid_argument("At least one exported V1 model is required");
    }
    std::vector<Model> models;
    models.reserve(paths.size());
    for (const auto& path : paths) {
        models.emplace_back(path);
    }
    return models;
}

double predict_models(const std::vector<Model>& models, const Features& input) {
    const Time time(input);
    double sum = 0;
    for (const auto& model : models) {
        sum += model.predict(input, time);
    }
    const double value = sum / models.size();
    if (!std::isfinite(value)) {
        throw std::runtime_error("V1 produced a non-finite dP prediction");
    }
    return value;
}

} // namespace

void predict_packed(const std::filesystem::path& input,
                    const std::vector<std::filesystem::path>& paths, std::ostream& out) {
    const auto models = load_models(paths);
    Reader src(input);
    src.magic("H9V1REQ1");
    const auto n = src.scalar<std::uint32_t>();
    if (n == 0) {
        src.fail("No records in V1 request");
    }
    for (std::uint32_t j = 0; j < n; ++j) {
        Features x;
        for (double& t : x.minutes) {
            t = src.scalar<double>();
            if (std::isinf(t)) {
                src.fail("Infinite scan time");
            }
        }
        for (std::size_t g = 0; g < dimensions.size(); ++g) {
            x.groups[g] = src.array<float>(frames * dimensions[g]);
            if (std::any_of(x.groups[g].begin(), x.groups[g].end(), [](float v) { return std::isinf(v); })) {
                src.fail("Infinite V1 descriptor");
            }
        }
        out << std::setprecision(17) << predict_models(models, x) << '\n';
    }
    src.end();
}

} // namespace h9dp::v1

namespace h9dp {

struct InferenceBest::Impl {
    explicit Impl(const std::vector<std::filesystem::path>& paths) : models(v1::load_models(paths)) {}
    std::vector<v1::Model> models;
};

InferenceBest::InferenceBest(const std::filesystem::path& path)
    : InferenceBest(std::vector<std::filesystem::path>{path}) {}

InferenceBest::InferenceBest(const std::vector<std::filesystem::path>& paths)
    : impl_(std::make_shared<Impl>(paths)) {}

double InferenceBest::predict(const std::vector<HimawariScan>& scans, StormCenter center) const {
    if (!impl_) {
        throw std::logic_error("InferenceBest has been moved from");
    }
    if (scans.empty() || scans.size() > v1::frames) {
        throw std::invalid_argument("V1 accepts one to six Himawari scans");
    }
    if (!std::isfinite(center.latitude_deg) || std::abs(center.latitude_deg) > 90
        || !std::isfinite(center.longitude_deg) || std::abs(center.longitude_deg) > 360) {
        throw std::invalid_argument("Invalid storm center latitude/longitude in degrees");
    }
    std::vector<v1::Crop> crops;
    crops.reserve(scans.size());
    for (const auto& scan : scans) {
        if (scan.minutes.has_value() != scans.front().minutes.has_value()
            || (scan.minutes && !std::isfinite(*scan.minutes))) {
            throw std::invalid_argument("Provide finite minute timestamps for every scan, or omit all of them");
        }
        crops.push_back(v1::load_crop(scan, center));
    }
    return v1::predict_models(impl_->models, v1::describe(std::move(crops)));
}

} // namespace h9dp
