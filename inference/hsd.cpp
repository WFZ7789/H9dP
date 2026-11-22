#include "v1_detail.hpp"

#ifdef H9DP_HAVE_BZIP2
#include <bzlib.h>
#endif
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace h9dp::v1 {
namespace {

using Bytes = std::vector<std::uint8_t>;
constexpr std::size_t max_file = 128 * 1024 * 1024;
constexpr double radians = 0.017453292519943295769;
constexpr std::array<int, 4> bands{8, 9, 10, 16};
constexpr int pixels = crop_size * crop_size;

Bytes read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto n = file.tellg();
    if (n <= 0 || n > static_cast<std::streamoff>(max_file)) {
        throw std::runtime_error("Unreadable HSD or invalid file size");
    }
    Bytes data(static_cast<std::size_t>(n));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), n)) {
        throw std::runtime_error("Failed to read HSD");
    }
    return data;
}

template<class Stream, class Step>
Bytes expand(Stream& stream, Step step, int ok, int end, std::size_t limit) {
    std::array<std::uint8_t, 65536> block{};
    Bytes out;
    int code;
    do {
        stream.next_out = reinterpret_cast<decltype(stream.next_out)>(block.data());
        stream.avail_out = static_cast<unsigned>(block.size());
        const auto before = stream.avail_in;
        code = step();
        const auto produced = block.size() - stream.avail_out;
        if ((code != ok && code != end) || (code == ok && produced == 0 && before == stream.avail_in)
            || produced > limit - out.size()) {
            throw std::runtime_error("Invalid or oversized compressed HSD stream");
        }
        out.insert(out.end(), block.begin(), block.begin() + produced);
    } while (code != end);
    if (stream.avail_in != 0) {
        throw std::runtime_error("Trailing bytes after compressed HSD stream");
    }
    return out;
}

Bytes decompress(Bytes& in, std::size_t offset, int method, std::size_t limit) {
    if (method == 1) {
        z_stream stream{};
        if (inflateInit2(&stream, 15 + 16) != Z_OK) {
            throw std::runtime_error("Cannot initialize HSD gzip decoder");
        }
        const std::unique_ptr<z_stream, decltype(&inflateEnd)> cleanup(&stream, inflateEnd);
        stream.next_in = in.data() + offset;
        stream.avail_in = static_cast<unsigned>(in.size() - offset);
        return expand(stream, [&] { return inflate(&stream, Z_NO_FLUSH); }, Z_OK, Z_STREAM_END, limit);
    }
    if (method == 2) {
#ifdef H9DP_HAVE_BZIP2
        bz_stream stream{};
        if (BZ2_bzDecompressInit(&stream, 0, 0) != BZ_OK) {
            throw std::runtime_error("Cannot initialize HSD bzip2 decoder");
        }
        const std::unique_ptr<bz_stream, decltype(&BZ2_bzDecompressEnd)> cleanup(&stream, BZ2_bzDecompressEnd);
        stream.next_in = reinterpret_cast<char*>(in.data() + offset);
        stream.avail_in = static_cast<unsigned>(in.size() - offset);
        return expand(stream, [&] { return BZ2_bzDecompress(&stream); }, BZ_OK, BZ_STREAM_END, limit);
#else
        throw std::runtime_error("This build has no BZip2 support");
#endif
    }
    throw std::runtime_error("Unknown HSD compression code");
}

// Assemble the bit pattern explicitly so DAT byte order is independent of the host.
template<class T>
T get(const Bytes& data, std::size_t i, bool little) {
    using Bits = std::conditional_t<sizeof(T) <= 4, std::uint32_t, std::uint64_t>;
    Bits bits = 0;
    for (std::size_t j = 0; j < sizeof(T); ++j) {
        bits = (bits << 8) | data[i + (little ? sizeof(T) - 1 - j : j)];
    }
    if constexpr (std::is_integral_v<T>) {
        return static_cast<T>(bits);
    } else {
        T value;
        std::memcpy(&value, &bits, sizeof(T));
        return value;
    }
}

struct Header {
    bool little;
    int band, width, height, first_line, compression;
    std::size_t header_size, data_size;
    std::string satellite, area;
    // Subsatellite longitude, factors/offsets, satellite distance, Earth radii.
    std::array<double, 8> geo;
    double mid_mjd;
    std::uint16_t invalid, outside;
    double wavelength, gain, intercept;
    std::array<double, 3> correction, constants;
};

Header parse(const Bytes& data) {
    static_assert(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    if (data.size() < 282 || data[0] != 1 || data[5] > 1) {
        throw std::runtime_error("Invalid HSD basic header");
    }
    Header h{};
    h.little = data[5] == 0;
    const auto u16 = [&](std::size_t p) { return get<std::uint16_t>(data, p, h.little); };
    const auto u32 = [&](std::size_t p) { return get<std::uint32_t>(data, p, h.little); };
    const auto f32 = [&](std::size_t p) { return get<float>(data, p, h.little); };
    const auto f64 = [&](std::size_t p) { return get<double>(data, p, h.little); };
    h.header_size = u32(70);
    h.data_size = u32(74);
    if (h.header_size < 282 || h.header_size > data.size() || h.data_size != data.size() - h.header_size) {
        throw std::runtime_error("HSD file size differs from header");
    }
    constexpr std::array<int, 11> minimum{282, 10, 51, 3, 107, 3, 7, 3, 3, 5, 3};
    std::array<std::size_t, 11> off{};
    std::size_t p = 0;
    for (std::size_t i = 0; i < off.size(); ++i) {
        if (h.header_size - p < (i == 9 ? 5U : 3U) || data[p] != i + 1) {
            throw std::runtime_error("Missing HSD header block");
        }
        const auto size = i == 9 ? u32(p + 1) : u16(p + 1);
        if (size < static_cast<unsigned>(minimum[i]) || size > h.header_size - p) {
            throw std::runtime_error("Invalid HSD header block size");
        }
        off[i] = p;
        p += size;
    }
    if (p != h.header_size) {
        throw std::runtime_error("HSD block lengths differ from header size");
    }
    const auto d = off[1], g = off[2], c = off[4], segment = off[6];
    h.width = u16(d + 5);
    h.height = u16(d + 7);
    h.compression = data[d + 9];
    h.band = u16(c + 3);
    h.first_line = u16(segment + 5);
    if (u16(d + 3) != 16 || h.width < crop_size || h.height < crop_size || h.first_line == 0
        || static_cast<std::size_t>(h.width) * h.height * 2 > max_file
        || std::find(bands.begin(), bands.end(), h.band) == bands.end()) {
        throw std::runtime_error("Unsupported HSD dimensions, bit depth, band or segment");
    }
    h.satellite.assign(data.begin() + 6, data.begin() + 22);
    h.area.assign(data.begin() + 38, data.begin() + 42);
    h.geo = {f64(g + 3), static_cast<double>(u32(g + 11)), static_cast<double>(u32(g + 15)),
             f32(g + 19), f32(g + 23), f64(g + 27), f64(g + 35), f64(g + 43)};
    const double start = f64(46), end = f64(54);
    h.mid_mjd = (start + end) / 2;
    h.wavelength = f64(c + 5) * 1e-6;
    h.invalid = u16(c + 15);
    h.outside = u16(c + 17);
    h.gain = f64(c + 19);
    h.intercept = f64(c + 27);
    for (std::size_t i = 0; i < 3; ++i) {
        h.correction[i] = f64(c + 35 + 8 * i);
        h.constants[i] = f64(c + 83 + 8 * i);
    }
    const auto positive = [](double x) { return std::isfinite(x) && x > 0; };
    const auto finite = [](double x) { return std::isfinite(x); };
    if (!std::all_of(h.geo.begin(), h.geo.end(), finite) || h.geo[1] == 0 || h.geo[2] == 0
        || h.geo[5] <= h.geo[6] || h.geo[6] < h.geo[7] || h.geo[7] <= 0
        || !finite(start) || !finite(end) || start < 0 || start > end || end > 1e6
        || !positive(h.wavelength) || !finite(h.gain) || !finite(h.intercept)
        || !std::all_of(h.correction.begin(), h.correction.end(), finite)
        || !std::all_of(h.constants.begin(), h.constants.end(), positive)) {
        throw std::runtime_error("Invalid HSD projection, timing or calibration");
    }
    return h;
}

struct Window {
    int x, y;
    double center_x, center_y;
};

Window window_for(const Header& h, StormCenter center) {
    const auto [lon0, cfac, lfac, coff, loff, distance, equatorial, polar] = h.geo;
    const double lat = center.latitude_deg * radians;
    const double lon = (center.longitude_deg - lon0) * radians;
    const double ratio = (polar / equatorial) * (polar / equatorial);
    const double phi = std::atan(ratio * std::tan(lat));
    const double c = std::cos(phi), s = std::sin(phi);
    const double radius = polar / std::sqrt(1 - (1 - ratio) * c * c);
    const double sx = radius * c * std::cos(lon), sy = radius * c * std::sin(lon), sz = radius * s;
    const double r1 = distance - sx;
    if (sx * r1 - sy * sy - sz * sz / ratio <= 0) {
        throw std::runtime_error("Storm center is not visible from this satellite");
    }
    const double col = coff + std::atan2(sy, r1) / radians * cfac / 65536 - 1;
    const double row = loff + std::asin(-sz / std::sqrt(r1 * r1 + sy * sy + sz * sz)) / radians * lfac / 65536
                       - h.first_line;
    if (!(col >= 0 && col < h.width && row >= 0 && row < h.height)) {
        throw std::runtime_error("Storm center lies outside this HSD segment");
    }
    const int x = std::clamp(static_cast<int>(std::floor(col + 0.5)) - crop_size / 2, 0, h.width - crop_size);
    const int y = std::clamp(static_cast<int>(std::floor(row + 0.5)) - crop_size / 2, 0, h.height - crop_size);
    return {x, y, col - x, row - y};
}

std::vector<std::uint8_t> earth(const Header& h, const Window& w) {
    const auto [lon0, cfac, lfac, coff, loff, distance, equatorial, polar] = h.geo;
    const double ratio = (equatorial / polar) * (equatorial / polar);
    const double disc2 = distance * distance - equatorial * equatorial;
    std::array<double, crop_size> cx{};
    for (int x = 0; x < crop_size; ++x) {
        cx[x] = std::cos((w.x + x + 1 - coff) * 65536 / cfac * radians);
    }
    std::vector<std::uint8_t> mask(pixels);
    for (int y = 0; y < crop_size; ++y) {
        const double phi = (w.y + y + h.first_line - loff) * 65536 / lfac * radians;
        const double c = std::cos(phi), s = std::sin(phi);
        const double right = (c * c + ratio * s * s) * disc2;
        for (int x = 0; x < crop_size; ++x) {
            const double left = distance * cx[x] * c;
            mask[y * crop_size + x] = left * left >= right;
        }
    }
    return mask;
}

std::vector<std::int16_t> calibrate(Bytes& data, const Header& h, const Window& w,
                                    std::vector<std::uint8_t>& valid) {
    const auto bytes = static_cast<std::size_t>(h.width) * h.height * 2;
    Bytes decoded;
    std::size_t offset = h.header_size;
    if (h.compression != 0) {
        decoded = decompress(data, offset, h.compression, bytes);
        offset = 0;
    }
    const Bytes& payload = h.compression ? decoded : data;
    if (payload.size() - offset != bytes) {
        throw std::runtime_error("HSD image payload has an incorrect size");
    }
    const auto [speed, planck, boltzmann] = h.constants;
    const auto [c0, c1, c2] = h.correction;
    const double a = (planck * speed) / (boltzmann * h.wavelength);
    const double b = (2 * planck * (speed * speed)) / std::pow(h.wavelength, 5);
    std::vector<std::int16_t> output(pixels, 10000); // 250 K placeholder in the V1 cache.
    for (int y = 0; y < crop_size; ++y) {
        for (int x = 0; x < crop_size; ++x) {
            const int i = y * crop_size + x;
            if (!valid[i]) {
                continue;
            }
            const auto p = offset + 2 * (static_cast<std::size_t>(w.y + y) * h.width + w.x + x);
            const auto count = get<std::uint16_t>(payload, p, h.little);
            const double rad = count * h.gain + h.intercept;
            if (count == h.invalid || count == h.outside || !(rad > 0)) {
                valid[i] = 0;
                continue;
            }
            const double t = a / std::log1p(b / (rad * 1e6));
            const float kelvin = static_cast<float>(c0 + c1 * t + c2 * (t * t));
            if (!std::isfinite(kelvin) || kelvin <= 0) {
                valid[i] = 0;
                continue;
            }
            const float q = std::nearbyint(kelvin * 100.0F - 15000.0F);
            if (q < -32768.0F || q > 32767.0F) {
                throw std::runtime_error("Brightness temperature outside int16 centi-Kelvin range");
            }
            output[i] = static_cast<std::int16_t>(q);
        }
    }
    return output;
}

} // namespace

Crop load_crop(const HimawariScan& scan, StormCenter center) {
    Crop crop{};
    std::optional<Header> first;
    Window w{};
    std::array<double, bands.size()> midpoint{};
    for (const auto& path : scan.files) {
        try {
            auto data = read_file(path);
            if (path.extension() == ".bz2" || path.extension() == ".gz") {
                data = decompress(data, 0, path.extension() == ".bz2" ? 2 : 1, max_file);
            }
            const auto h = parse(data);
            const auto channel = static_cast<std::size_t>(std::find(bands.begin(), bands.end(), h.band) - bands.begin());
            if (!crop.temperature[channel].empty()) {
                throw std::runtime_error("Duplicate HSD band in one scan");
            }
            if (!first) {
                first = h;
                w = window_for(h, center);
                crop.x = w.center_x;
                crop.y = w.center_y;
                crop.valid = earth(h, w);
            } else if (h.geo != first->geo || h.width != first->width || h.height != first->height
                       || h.first_line != first->first_line || h.satellite != first->satellite || h.area != first->area) {
                throw std::runtime_error("HSD bands of a scan must share the satellite and geometry");
            }
            midpoint[channel] = h.mid_mjd;
            crop.temperature[channel] = calibrate(data, h, w, crop.valid);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(path.string() + ": " + e.what());
        }
    }
    double minutes = 0;
    // Match preparation's check against B08 regardless of the caller's band order.
    for (double t : midpoint) {
        if (std::abs(t - midpoint[0]) * 1440 > 1.000001) {
            throw std::runtime_error(scan.files.front().string() + ": HSD band times differ from B08 by over one minute");
        }
        minutes += t * 1440 / bands.size();
    }
    for (auto& band : crop.temperature) {
        for (int i = 0; i < pixels; ++i) {
            if (!crop.valid[i]) {
                band[i] = 10000;
            }
        }
    }
    crop.minutes = scan.minutes.value_or(minutes);
    return crop;
}

} // namespace h9dp::v1
