#include <iostream>
#include <iomanip>
#include <fstream>
#include <bit>
#include <cstdint>

int main() {
    std::ifstream infile("input.txt");
    if (!infile) {
        std::cerr << "Error: Cannot open input file." << std::endl;
        return 1;
    }

    std::uint32_t a, b, i;
    float f;

    while (infile >> a >> b) {
        float fa = std::bit_cast<float>(a);
        float fb = std::bit_cast<float>(b);
        f = fa + fb;
        i = std::bit_cast<std::uint32_t>(f);
        std::cout << std::right << std::setw(10) << i << std::endl;
    }

    return 0;
}