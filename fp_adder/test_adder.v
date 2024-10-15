`include "fpadder.v"

module test_adder;
    // Inputs and outputs for the adder
    reg [31:0] a, b;
    wire [31:0] result;

    // Instantiating the adder
    fp32_adder adder(
        .a(a),
        .b(b),
        .sum(result)
    );

    // File descriptors
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
            code = $fscanf(in_file, "%d %d\n", in_data_a, in_data_b);
            if (code == 2) begin  // Successful read
                a = in_data_a;
                b = in_data_b;

                // Wait for the result
                #1;

                // Output the result
                $display("%d", result);
            end
        end

        $fclose(in_file);
        // $finish;
    end
endmodule
