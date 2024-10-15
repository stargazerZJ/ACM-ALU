`include "fpadder.v"

module test_adder;
	reg [7:0] a, b;
	wire [7:0] answer;
    wire [19:0] a_int, b_int;

	fp8_e4m3_adder adder (a, b, answer);

    fp8_to_int u_fp8_to_int_a(.fp8(a), .int_val(a_int));
    fp8_to_int u_fp8_to_int_b(.fp8(b), .int_val(b_int));

	integer i;

	initial begin
		$display("Testing your adder...");
		for(i=1; i<=100; i=i+1) begin
			a[7:0] = $random;
			b[7:0] = $random;

            #1;

			// $display("TESTCASE %d: %x + %x = %x", i, a, b, answer);
			$display("TESTCASE %d: %x + %x = %x, int %d %d", i, a, b, answer, a_int, b_int);
		end
	end
endmodule
