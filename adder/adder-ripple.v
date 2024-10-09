/* ACM Class System (I) Fall Assignment 1
 *
 *
 * Implement your ripple adder here
 *
 * GUIDE:
 *   1. Create a RTL project in Vivado
 *   2. Put this file into `Sources'
 *   3. Put `test_adder.v' into `Simulation Sources'
 *   4. Run Behavioral Simulation
 *   5. Make sure to run at least 100 steps during the simulation (usually 100ns)
 *   6. You can see the results in `Tcl console'
 *
 */

module adder(
	input       [15:0]          a,
    input       [15:0]          b,
    output reg  [15:0]          sum,
	output reg  carry
);
	wire [15:0] P, G;
	wire [16:0] C;

	assign P = a ^ b;
	assign G = a & b;

	assign C[0] = 1'b0;

	generate
		genvar i;
		for (i = 0; i < 16; i = i + 1) begin : gen
			assign C[i+1] = (G[i] | (P[i] & C[i]));
		end
	endgenerate

	always @(*) begin
		sum = P ^ C[15:0];
		carry = C[16];
	end

endmodule
