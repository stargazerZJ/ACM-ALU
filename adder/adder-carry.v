/* ACM Class System (I) Fall Assignment 1
 *
 *
 * Implement your naive adder here
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


module carry_lookahead_adder_4bit(
        input [3:0] a,
        input [3:0] b,
        input cin,
        output [3:0] sum,
        // output cout
		output G,
		output P
    );

    wire [3:1] carry;
    wire [3:0] g;
    wire [3:0] p;

    assign g = a & b;
	assign p = a ^ b;

    assign carry[1] = g[0] ^ (p[0] & cin);
    assign carry[2] = g[1] ^ (p[1] & g[0]) ^ (p[1] & p[0] & cin);
    assign carry[3] = g[2] ^ (p[2] & g[1]) ^ (p[2] & p[1] & g[0]) ^ (p[2] & p[1] & p[0] & cin);
    // assign cout = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]) ^ (p[3] & p[2] & p[1] & p[0] & cin);
    assign sum = a ^ b ^ {carry[3:1], cin};
	assign G = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]);
	assign P = p[3] & p[2] & p[1] & p[0];


endmodule


module adder(
	input   [15:0]          a,
    input   [15:0]          b,
    output  [15:0]          sum,
	output  cout
);
	wire [3:1] carry;
	wire [3:0] g;
	wire [3:1] p;

	carry_lookahead_adder_4bit cla0 (
		.a(a[3:0]),
		.b(b[3:0]),
		.cin(1'b0),      // Initial carry-in is 0
		.sum(sum[3:0]),
		.G(g[0])
		// .P(p[0]) // p[0] is not used
	);

	// assign carry[1] = g[0] ^ (p[0] & 1'b0);
	assign carry[1] = g[0]; // cin = 0

	carry_lookahead_adder_4bit cla1 (
		.a(a[7:4]),
		.b(b[7:4]),
		.cin(carry[1]),        // Carry-in from previous stage
		.sum(sum[7:4]),
		.G(g[1]),
		.P(p[1])
	);

	// assign carry[2] = g[1] ^ (p[1] & g[0]) ^ (p[1] & p[0] & 1'b0);
	assign carry[2] = g[1] ^ (p[1] & g[0]);

	carry_lookahead_adder_4bit cla2 (
		.a(a[11:8]),
		.b(b[11:8]),
		.cin(carry[2]),        // Carry-in from previous stage
		.sum(sum[11:8]),
		.G(g[2]),
		.P(p[2])
	);

	// assign carry[3] = g[2] ^ (p[2] & g[1]) ^ (p[2] & p[1] & g[0]) ^ (p[2] & p[1] & p[0] & 1'b0);
	assign carry[3] = g[2] ^ (p[2] & g[1]) ^ (p[2] & p[1] & g[0]);

	carry_lookahead_adder_4bit cla3 (
		.a(a[15:12]),
		.b(b[15:12]),
		.cin(carry[3]),        // Carry-in from previous stage
		.sum(sum[15:12]),
		.G(g[3]),
		.P(p[3])
	);

	// assign cout = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]) ^ (p[3] & p[2] & p[1] & p[0] & 1'b0);
	assign cout = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]);

endmodule
