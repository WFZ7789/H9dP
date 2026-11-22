#include "v1_detail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace h9dp::v1 {
namespace {

constexpr std::array<int, 18> radii{1, 2, 3, 5, 7, 10, 14, 19, 25, 33, 44, 58, 76, 98, 126, 160, 198, 228};
constexpr int nr = static_cast<int>(radii.size());
constexpr int angles = 96;
constexpr int channels = 4;
constexpr double step = 6.2831853071795864769 / angles;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
using Circle = std::array<double, angles>;
using Rings = std::array<std::array<Circle, nr>, channels>;
using Patch = std::vector<double>;
constexpr int half = 64, side = 2 * half + 1;

int reflect(int i) {
    return i < 0 ? -i - 1 : (i >= side ? 2 * side - i - 1 : i);
}

// scipy.ndimage.gaussian_filter's default 4-sigma truncation and reflect boundary.
Patch gaussian(const Patch& x, double sigma) {
    const int radius = static_cast<int>(4 * sigma + 0.5);
    std::vector<double> kernel(2 * radius + 1);
    double sum = 0;
    for (int k = -radius; k <= radius; ++k) {
        sum += kernel[k + radius] = std::exp(-0.5 * k * k / (sigma * sigma));
    }
    for (double& v : kernel) {
        v /= sum;
    }
    Patch tmp(side * side), y(side * side);
    for (int i = 0; i < side; ++i) {
        for (int k = -radius; k <= radius; ++k) {
            for (int j = 0; j < side; ++j) {
                tmp[i * side + j] += x[reflect(i + k) * side + j] * kernel[k + radius];
            }
        }
    }
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            for (int k = -radius; k <= radius; ++k) {
                y[i * side + j] += tmp[i * side + reflect(j + k)] * kernel[k + radius];
            }
        }
    }
    return y;
}

std::array<int, 2> warm_core(const Crop& crop) {
    const int x = static_cast<int>(std::nearbyint(crop.x - 0.5));
    const int y = static_cast<int>(std::nearbyint(crop.y - 0.5));
    Patch valid(side * side), warm(side * side);
    for (int i = 0; i < side; ++i) {
        const int row = y + i - half;
        if (row < 0 || row >= crop_size) {
            continue;
        }
        for (int j = 0; j < side; ++j) {
            const int col = x + j - half;
            if (col < 0 || col >= crop_size || !crop.valid[row * crop_size + col]) {
                continue;
            }
            const int p = row * crop_size + col;
            valid[i * side + j] = 1;
            warm[i * side + j] = (crop.temperature[2][p] + crop.temperature[3][p]) * 0.005;
        }
    }
    auto support = valid;
    std::array<Patch, 4> filtered;
    constexpr std::array<double, 4> sigmas{1.5, 4, 9, 18};
    for (std::size_t i = 0; i < sigmas.size(); ++i) {
        const auto w = gaussian(valid, sigmas[i]);
        filtered[i] = gaussian(warm, sigmas[i]);
        for (std::size_t p = 0; p < w.size(); ++p) {
            support[p] = support[p] && w[p] > 0.85;
            filtered[i][p] /= std::max(w[p], 1e-8);
        }
    }
    double best = -std::numeric_limits<double>::infinity();
    std::array<int, 2> result{x, y};
    for (int i = -22; i <= 22; ++i) {
        for (int j = -22; j <= 22; ++j) {
            const int p = (i + half) * side + j + half, r2 = i * i + j * j;
            if (r2 > 22 * 22 || !support[p]) {
                continue;
            }
            const double base = filtered[3][p];
            const double score = ((filtered[0][p] - base) + (filtered[1][p] - base) + (filtered[2][p] - base)) / 3
                                 - 0.009 * r2;
            if (score > best) {
                best = score;
                result = {x + j, y + i};
            }
        }
    }
    return result;
}

Rings polar_scan(const Crop& crop) {
    const auto [x, y] = warm_core(crop);
    Rings ring{};
    for (int r = 0; r < nr; ++r) {
        for (int a = 0; a < angles; ++a) {
            const double u = x + radii[r] * std::cos(a * step), v = y + radii[r] * std::sin(a * step);
            if (u < 0 || u > crop_size - 1 || v < 0 || v > crop_size - 1) {
                for (auto& band : ring) {
                    band[r][a] = missing;
                }
                continue;
            }
            const int c = static_cast<int>(u), l = static_cast<int>(v);
            const int c1 = std::min(c + 1, crop_size - 1), l1 = std::min(l + 1, crop_size - 1);
            const double wx0 = 1 - (u - c), wx1 = 1 - wx0, wy0 = 1 - (v - l), wy1 = 1 - wy0;
            const std::array<double, 4> w{wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1};
            const std::array<int, 4> p{l * crop_size + c, l * crop_size + c1, l1 * crop_size + c, l1 * crop_size + c1};
            const auto sample = [&](const auto& values) {
                return static_cast<float>(values[p[0]] * w[0] + values[p[1]] * w[1]
                                          + values[p[2]] * w[2] + values[p[3]] * w[3]);
            };
            const bool valid = sample(crop.valid) >= 0.999F;
            for (int b = 0; b < channels; ++b) {
                ring[b][r][a] = valid ? sample(crop.temperature[b]) * 0.01F + 150.0F : missing;
            }
        }
    }
    return ring;
}

double quantile(const double* x, int size, double q) {
    const double pos = (size - 1) * q;
    const int lo = static_cast<int>(pos), hi = std::min(lo + 1, size - 1);
    return x[lo] + (x[hi] - x[lo]) * (pos - lo);
}

struct Fourier {
    std::array<Circle, 16> cosine{}, sine{};
    Fourier() {
        for (int k = 1; k < 16; ++k) {
            for (int a = 0; a < angles; ++a) {
                cosine[k][a] = std::cos(k * a * step);
                sine[k][a] = std::sin(k * a * step);
            }
        }
    }
};

void texture(const Rings& ring, const std::array<float*, dimensions.size()>& row) {
    std::array<std::array<std::array<double, 5>, nr>, 8> quant{};
    for (int s = 0; s < 8; ++s) {
        for (int r = 0; r < nr; ++r) {
            Circle values;
            int count = 0;
            for (int a = 0; a < angles; ++a) {
                const double value = s < 4 ? ring[s][r][a]
                    : s < 7 ? ring[s - 3][r][a] - ring[s - 4][r][a] : ring[3][r][a] - ring[0][r][a];
                if (std::isfinite(value)) {
                    values[count++] = value;
                }
            }
            auto& q = quant[s][r];
            q.fill(missing);
            if (count >= angles * 0.65) {
                std::sort(values.begin(), values.begin() + count);
                constexpr std::array<double, 5> levels{0.03, 0.2, 0.5, 0.8, 0.97};
                for (int i = 0; i < 5; ++i) {
                    q[i] = quantile(values.data(), count, levels[i]);
                }
            }
            const auto g = r < 8 ? eye : wall;
            const int i = r < 8 ? s * 8 + r : s * 10 + r - 8;
            std::copy(q.begin(), q.end(), row[g] + i * 5);
        }
        const double base = (quant[s][9][0] + quant[s][10][0] + quant[s][11][0]) / 3;
        for (int r = 0; r < 8; ++r) {
            for (int i = 0; i < 5; ++i) {
                row[relief][(s * 8 + r) * 5 + i] = static_cast<float>(quant[s][r][i] - base);
            }
        }
    }
    static const Fourier harmonics;
    for (int b = 0; b < channels; ++b) {
        for (int r = 0; r < nr; ++r) {
            const auto& x = ring[b][r];
            if (std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); })) {
                std::array<double, 4> power{};
                for (int k = 1; k < 16; ++k) {
                    double re = 0, im = 0;
                    for (int a = 0; a < angles; ++a) {
                        re += x[a] * harmonics.cosine[k][a];
                        im += x[a] * harmonics.sine[k][a];
                    }
                    const int i = k < 2 ? 0 : k < 4 ? 1 : k < 8 ? 2 : 3;
                    power[i] += (re * re + im * im) / (angles * angles);
                }
                for (int i = 0; i < 4; ++i) {
                    row[waves][(b * nr + r) * 4 + i] = static_cast<float>(std::sqrt(power[i]));
                }
            }
            if (r >= 4) {
                for (int a = 0; a < 12; ++a) {
                    double sum = 0;
                    for (int j = 0; j < 8; ++j) {
                        sum += x[a * 8 + j];
                    }
                    row[sector][(b * (nr - 4) + r - 4) * 12 + a] = static_cast<float>(sum / 8);
                }
            }
        }
    }
}

// Equivalent circular correlation to irfft(rfft(b) * conj(rfft(a))) at the allowed shifts.
void motion_features(const Rings& a, const Rings& b, double minutes, float* out) {
    for (int r = 2; r < nr - 1; ++r) {
        const int limit = static_cast<int>(std::clamp(0.55 * minutes / (step * radii[r]), 2.0, angles / 3.0));
        Circle corr{};
        int used = 0;
        for (int c = 0; c < channels; ++c) {
            for (int j = r - 1; j <= r + 1; ++j) {
                const auto& xs = a[c][j];
                const auto& ys = b[c][j];
                double xm = 0, ym = 0;
                for (int p = 0; p < angles; ++p) {
                    xm += xs[p];
                    ym += ys[p];
                }
                if (!std::isfinite(xm) || !std::isfinite(ym)) {
                    continue;
                }
                xm /= angles;
                ym /= angles;
                Circle x, y;
                double xx = 0, yy = 0;
                for (int p = 0; p < angles; ++p) {
                    x[p] = xs[p] - xm;
                    y[p] = ys[p] - ym;
                    xx += x[p] * x[p];
                    yy += y[p] * y[p];
                }
                const double norm = std::sqrt(xx * yy);
                if (!(norm > angles * (0.35 * 0.35))) {
                    continue;
                }
                ++used;
                for (int s = -limit; s <= limit; ++s) {
                    double dot = 0;
                    for (int p = 0; p < angles; ++p) {
                        dot += x[p] * y[(p + s + angles) % angles];
                    }
                    corr[s + limit] += dot / norm;
                }
            }
        }
        if (!used) {
            continue;
        }
        const int best = static_cast<int>(std::max_element(corr.begin(), corr.begin() + 2 * limit + 1) - corr.begin());
        std::array<double, channels * angles> change;
        int n = 0;
        for (int c = 0; c < channels; ++c) {
            for (int p = 0; p < angles; ++p) {
                const double delta = b[c][r][p] - a[c][r][p];
                if (std::isfinite(delta)) {
                    change[n++] = delta;
                }
            }
        }
        float* row = out + (r - 2) * 4;
        row[0] = static_cast<float>((best - limit) * step * radii[r] / minutes);
        row[1] = static_cast<float>(corr[best] / used);
        row[2] = static_cast<float>((corr[best] - corr[limit]) / used);
        if (n > 0) {
            std::sort(change.begin(), change.begin() + n);
            row[3] = static_cast<float>(quantile(change.data(), n, 0.5) * 60 / minutes);
        }
    }
}

} // namespace

Features describe(std::vector<Crop> crops) {
    std::sort(crops.begin(), crops.end(), [](const Crop& a, const Crop& b) { return a.minutes < b.minutes; });
    Features out;
    out.minutes.fill(missing);
    for (std::size_t g = 0; g < dimensions.size(); ++g) {
        out.groups[g].resize(frames * dimensions[g], static_cast<float>(missing));
    }
    std::optional<Rings> previous;
    double last_time = 0;
    for (std::size_t i = 0; i < crops.size(); ++i) {
        const auto& crop = crops[i];
        if (i > 0 && crop.minutes <= crops[i - 1].minutes) {
            throw std::invalid_argument("Himawari scan times must be distinct");
        }
        out.minutes[i] = crop.minutes;
        if (std::none_of(crop.valid.begin(), crop.valid.end(), [](auto v) { return v != 0; })) {
            continue;
        }
        const auto ring = polar_scan(crop);
        std::array<float*, dimensions.size()> row;
        for (std::size_t g = 0; g < dimensions.size(); ++g) {
            row[g] = out.groups[g].data() + i * dimensions[g];
        }
        texture(ring, row);
        const double dt = crop.minutes - last_time;
        if (previous && dt > 0 && dt <= 130) {
            motion_features(*previous, ring, dt, row[motion]);
        }
        previous = ring;
        last_time = crop.minutes;
    }
    return out;
}

} // namespace h9dp::v1
