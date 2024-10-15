#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <limits>

// Utility to decode FP8 e4m3 from uint8_t
double fp8ToDouble(uint8_t fp8) {
    uint8_t sign = (fp8 >> 7) & 0x1;
    uint8_t exponent = (fp8 >> 3) & 0xF;
    uint8_t mantissa = fp8 & 0x7;

    if (fp8 == 0xFF || fp8 == 0x7F) {
        // NaN case
        return std::numeric_limits<double>::quiet_NaN();
    }

    double result;
    if (exponent == 0) {
        // Subnormal number
        result = mantissa * std::pow(2, -6 - 3);  // Subnormals have an implicit exponent of -6
    } else {
        // Normalized number
        int exp_val = exponent - 7;  // Apply the bias
        result = (1.0 + mantissa / 8.0) * std::pow(2, exp_val);
    }

    return sign ? -result : result;
}

// Utility to encode double to FP8 e4m3 format
uint8_t doubleToFp8(double val) {
    if (std::isnan(val)) {
        // Encode NaN
        return 0x7F;  // S.1111.111 encoding for NaN
    }

    uint8_t sign = val < 0 ? 0x80 : 0x00;  // 1 bit sign
    val = std::fabs(val);  // Work with absolute value

    if (val >= 448.0) {
        // Overflow, saturate to max FP8 value (0x7E for positive or 0xFE for negative)
        return sign | 0x7E;
    } else if (val < std::pow(2, -9)) {
        // Too small for subnormals (underflow)
        return sign;
    }

    // Find an exponent that represents val
    int exp;
    double mantissa = std::frexp(val, &exp);

    // Convert exponent to FP8 format (bias by 7)
    int biased_exp = exp + 6;

    if (biased_exp <= 0) {
        // Subnormal number
        uint8_t sub_mantissa = round(std::ldexp(val, 6) * 8);
        return sign | sub_mantissa;
    } else if (biased_exp > 15) {
        // Saturate in case of overflow to largest representable value
        return sign | 0x7E;
    } else {
        // Normalized number, construct FP8 value
        uint8_t normalized_mantissa = floor(mantissa * 16);
        // bool round_up = round(mantissa * 16) != normalized_mantissa;
        bool round_up;
        if (normalized_mantissa & 1) {
            // Round to nearest even
            round_up = (mantissa * 16 - normalized_mantissa) >= 0.5;
        } else {
            round_up = (mantissa * 16 - normalized_mantissa) > 0.5;
        }
        if ((normalized_mantissa + round_up) == 16) {
            // Overflow, round up the exponent
            biased_exp++;
            if (biased_exp > 15) {
                // Saturate in case of overflow to largest representable value
                return sign | 0x7E;
            }
        }
        return sign | (biased_exp << 3) | ((normalized_mantissa + round_up) & 0x7);
    }
}

// FP8 E4M3 adder
uint8_t fp8Add(uint8_t a, uint8_t b) {
    double da = fp8ToDouble(a);
    double db = fp8ToDouble(b);

    double result = da + db;

    // Convert the result back to FP8
    uint8_t fp8_result = doubleToFp8(result);

    double dr = fp8ToDouble(fp8_result);

    std::cerr << "a: " << std::hex << static_cast<int>(a) << " (" << da << ")" << std::endl;
    std::cerr << "b: " << std::hex << static_cast<int>(b) << " (" << db << ")" << std::endl;
    std::cerr << "result: " << std::hex << static_cast<int>(fp8_result) << " (" << dr << "), Actual: " << result << std::endl;

    return fp8_result;
}

int main() {
    unsigned int a, b;
    uint8_t result;

    while (std::cin >> std::hex >> a >> std::hex >> b) {
        result = fp8Add(a, b);
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(result) << std::endl;
    }

    return 0;
}