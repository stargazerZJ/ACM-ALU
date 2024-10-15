`include "fpadder.v"

module test_adder;
	reg [7:0] a, b;
	wire [7:0] answer;
    wire [19:0] a_int, b_int;

	fp8_e4m3_adder adder (a, b, answer);

    fp8_to_int u_fp8_to_int_a(.fp8(a), .int_val(a_int));
    fp8_to_int u_fp8_to_int_b(.fp8(b), .int_val(b_int));

    integer in_file, code;
    reg [31:0] in_data_a;
    reg [31:0] in_data_b;

    initial begin
        // Open input
        in_file = $fopen("input.txt", "r");
        if (in_file == 0) begin
            $display("Error: Failed to open input.");
            $finish;
        end

        // Read loop
        while (!$feof(in_file)) begin
            // Read two 32-bit unsigned integers
            code = $fscanf(in_file, "%x %x\n", in_data_a, in_data_b);
            if (code == 2) begin  // Successful read
                a = in_data_a;
                b = in_data_b;

                // Wait for the result
                #1;

                // Output the result
                $display("%x", answer);
				// $display("%x + %x = %x, int %d %d", a, b, answer, a_int, b_int);
            end
        end

        $fclose(in_file);
        // $finish;
    end
endmodule
