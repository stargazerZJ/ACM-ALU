/* ACM Class System (I) Fall Assignment 1
 *
 *
 * This file is used to test your adder.
 * Please DO NOT modify this file.
 *
 * GUIDE:
 *   1. Create a RTL project in Vivado
 *   2. Put `adder.v' OR `adder2.v' into `Sources', DO NOT add both of them at the same time.
 *   3. Put this file into `Simulation Sources'
 *   4. Run Behavioral Simulation
 *   5. Make sure to run at least 100 steps during the simulation (usually 100ns)
 *   6. You can see the results in `Tcl console'
 *
 */

`include "adder.v"

module test_adder;
	wire [15:0] answer;
	wire carry;
	reg [15:0] a, b;
	reg [16:0] res;

	adder adder (a, b, answer, carry);

	integer i;
	integer error_count = 0; // Counter for wrong answers

	initial begin
		$display("Testing your adder...");
		for(i=1; i<=100; i=i+1) begin
			a[15:0] = $random;
			b[15:0] = $random;
			res = a + b;

			#1;
			if (answer !== res[15:0] || carry != res[16]) begin
				$display("TESTCASE %d: %d + %d = %d carry: %d (Expected: %d carry: %d) - Wrong Answer!",
					i, a, b, answer, carry, res[15:0], res[16]);
				error_count = error_count + 1;
			end else begin
				// $display("TESTCASE %d: %d + %d = %d carry: %d - Correct", i, a, b, answer, carry);
			end
		end

		if (error_count == 0) begin
			$display("Congratulations! You have passed all of the tests.");
			$finish(0); // Correct exiting code for all tests passed
		end else begin
			$error("You have %d failing tests.", error_count);
			$finish(1); // Error exiting code for one or more tests failed
		end
	end
endmodule
