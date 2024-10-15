
module fp8_to_int(
    input  [7:0]   fp8,
    output [19:0]  int_val
);
    wire sign;
    wire [3:0] exponent;
    wire [3:0] exponent_adjusted;
    wire [2:0] mantissa;
    wire [3:0] mantissa_ext;
    wire [19:0] adjusted_mantissa;

    assign sign = fp8[7];
    assign exponent = fp8[6:3];
    assign mantissa = fp8[2:0];

    // Check for subnormal case: exponent is 0
    wire is_subnormal = (exponent == 4'b0000);

    // Extend mantissa with leading 1 for normal, and leading 0 for subnormals
    assign mantissa_ext = is_subnormal ? {1'b0, mantissa}
                                       : {1'b1, mantissa};  // Leading 1 for normal values

    // Adjust exponent for 2^9 scaling and -7 bias
    assign exponent_adjusted = is_subnormal ? 4'b0001 : exponent;  // For subnormal, shift by 1
    assign adjusted_mantissa = ({16'b0, mantissa_ext} << (exponent_adjusted - 1));  // 9 - 7 - 3 = -1

    // Apply sign
    assign int_val = sign ? -adjusted_mantissa : adjusted_mantissa;

endmodule

module normalize (
    input [18:0] abs_val,
    output reg [4:0] leading_zeros, // range: 2 - 17
    output reg [7:0] normalized_val // zero-padded 8-bit value
);
    reg [7:0] normalized;

    always @(*) begin
        // Initialize leading zeros
        leading_zeros = 0;
        normalized_val = 0;

        // Detect leading zeros (2 - 17 range).

        // Only considering the 18 most significant bits of abs_val (abs_val[17:0])
        if (abs_val[17])     leading_zeros = 2;
        else if (abs_val[16]) leading_zeros = 3;
        else if (abs_val[15]) leading_zeros = 4;
        else if (abs_val[14]) leading_zeros = 5;
        else if (abs_val[13]) leading_zeros = 6;
        else if (abs_val[12]) leading_zeros = 7;
        else if (abs_val[11]) leading_zeros = 8;
        else if (abs_val[10]) leading_zeros = 9;
        else if (abs_val[9]) leading_zeros = 10;
        else if (abs_val[8])  leading_zeros = 11;
        else if (abs_val[7])  leading_zeros = 12;
        else if (abs_val[6])  leading_zeros = 13;
        else if (abs_val[5])  leading_zeros = 14;
        else if (abs_val[4])  leading_zeros = 15;
        else if (abs_val[3])  leading_zeros = 16;
        else                  leading_zeros = 17; // Subnormal case

        // Normalize the abs_val by shifting left based on the detected leading zeros
        // normalized_val = (abs_val[17:0] >> (12 - leading_zeros));
        normalized = (leading_zeros <= 12) ?
                 (abs_val[17:0] >> (12 - leading_zeros)) :
                 (abs_val[17:0] << (leading_zeros - 12));
        normalized_val = (leading_zeros == 17) ? normalized >> 1 : normalized;
    end

endmodule

module int_to_fp8 (
    input [19:0] int_val,
    output reg [7:0] fp8_val
);

    wire sign;
    wire [18:0] abs_val;
    wire [4:0] leading_zeros;
    wire [3:0] exponent;
    wire [3:0] exponent_adjusted;
    wire [2:0] mantissa;
    wire [7:0] normalized_val;

    assign sign = int_val[19];
    assign abs_val = sign ? -int_val[18:0] : int_val[18:0];

    // Normalization module
    normalize norm_inst (
        .abs_val(abs_val),
        .leading_zeros(leading_zeros),
        .normalized_val(normalized_val)
    );

    // Calculate exponent and mantissa
    assign exponent = 5'd17 - leading_zeros; // 20 - leading_zeros - 1 - 9 + 7 = 17
    assign exponent_adjusted = exponent == 4'd0 ? 4'd0001 : exponent;  // For subnormal, shift by 1
    assign mantissa = normalized_val[6:4];

    // Rounding (round to nearest, ties to even)
    wire round_up = normalized_val[3] & (normalized_val[2] | normalized_val[1] | normalized_val[0] | mantissa[0]);

    // Saturation and final FP8 value assignment
    always @(*) begin
        // if (abs_val[19:18] != 2'b00 || abs_val >> 15 == 7) begin
        //     fp8_val = {sign, 7'b1111110}; // Saturate to max value
        // end else begin
        //     fp8_val = abs_val >> 12;
        // end
        if (int_val == 20'd0) begin
            fp8_val = 8'd0; // Zero
        // end else if (abs_val[19:18] != 2'b00 || abs_val[19:15] == 5'b00111) begin
        end else if (abs_val[19:18] != 2'b00 || abs_val >> 15 == 7) begin
            fp8_val = {sign, 7'b1111110}; // Saturate to max value
        end else if (round_up && mantissa == 3'b111) begin
            fp8_val = {sign, exponent[3:0] + 4'd1, 3'b000}; // Overflow to next exponent
        end else begin
            fp8_val = {sign, exponent[3:0], mantissa + round_up};
        end
    end

endmodule

module fp8_e4m3_adder(
    input  [7:0] a,  // 8-bit FP8 number a
    input  [7:0] b,  // 8-bit FP8 number b
    output [7:0] sum // 8-bit FP8 sum
);
    wire a_is_nan, b_is_nan;

    // Deal with NaN values; nan is represented by both exponent and mantissa being all 1s, different from IEEE 754
    assign a_is_nan = (a[6:0] == 7'b1111111);
    assign b_is_nan = (b[6:0] == 7'b1111111);

    // If either input is NaN, return NaN
    wire [7:0] nan_value = 8'b01111111;  // NaN representation for FP8

    wire [19:0] a_int, b_int;
    wire [19:0] sum_int;

    // Convert FP8 to integer (20-bit), handle subnormals
    fp8_to_int u_fp8_to_int_a(.fp8(a), .int_val(a_int));
    fp8_to_int u_fp8_to_int_b(.fp8(b), .int_val(b_int));

    // Add the two integer values
    assign sum_int = a_int + b_int;

    wire [7:0] sum_fp8;
    // Convert back from the integer sum to FP8, truncate overflow
    int_to_fp8 u_int_to_fp8(.int_val(sum_int), .fp8_val(sum_fp8));

    // Decide the output: NaN or calculated sum
    assign sum = (a_is_nan || b_is_nan) ? nan_value : sum_fp8;

endmodule