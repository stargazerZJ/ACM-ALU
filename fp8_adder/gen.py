import random

initial_data = [
    "24 81",    # Standard case
    "09 63",
    "a2 68",
    "1a 2a",
    "61 bf",
    "0d 8d",    # Zero case
    "00 00",
    "f9 f9",    # Overflow case
    "79 79",
    "fd e8",
    "12 8f",    # Subnormal case
    "7f 00",    # NaN case
    "ff 00",
    "ff ff"
]

def generate_hex_pair():
    # Generate two random uint8 numbers and convert to 2-character hex values
    hex1 = f"{random.randint(0, 255):02x}"
    hex2 = f"{random.randint(0, 255):02x}"
    return hex1, hex2

def generate_hex_pairs(n):
    for _ in range(n):
        hex1, hex2 = generate_hex_pair()
        print(f"{hex1} {hex2}")

# Print initial data
for data in initial_data:
    print(data)

# Generate 1000 hex pairs
generate_hex_pairs(1000)