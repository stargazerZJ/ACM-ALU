

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

module carry_lookahead_adder_16bit(
        input [15:0] a,
        input [15:0] b,
        input cin,
        output [15:0] sum,
        // output cout
		output G,
		output P
    );

	wire [3:1] carry;
	wire [3:0] g;
	wire [3:0] p;

	carry_lookahead_adder_4bit cla0 (
		.a(a[3:0]),
		.b(b[3:0]),
		.cin(cin),
		.sum(sum[3:0]),
		.G(g[0]),
		.P(p[0])
	);

	assign carry[1] = g[0] ^ (p[0] & cin);

	carry_lookahead_adder_4bit cla1 (
		.a(a[7:4]),
		.b(b[7:4]),
		.cin(carry[1]),        // Carry-in from previous stage
		.sum(sum[7:4]),
		.G(g[1]),
		.P(p[1])
	);

	assign carry[2] = g[1] ^ (p[1] & g[0]) ^ (p[1] & p[0] & cin);

	carry_lookahead_adder_4bit cla2 (
		.a(a[11:8]),
		.b(b[11:8]),
		.cin(carry[2]),        // Carry-in from previous stage
		.sum(sum[11:8]),
		.G(g[2]),
		.P(p[2])
	);

	assign carry[3] = g[2] ^ (p[2] & g[1]) ^ (p[2] & p[1] & g[0]) ^ (p[2] & p[1] & p[0] & cin);

	carry_lookahead_adder_4bit cla3 (
		.a(a[15:12]),
		.b(b[15:12]),
		.cin(carry[3]),        // Carry-in from previous stage
		.sum(sum[15:12]),
		.G(g[3]),
		.P(p[3])
	);

	assign cout = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]) ^ (p[3] & p[2] & p[1] & p[0] & cin);
    assign G = g[3] ^ (p[3] & g[2]) ^ (p[3] & p[2] & g[1]) ^ (p[3] & p[2] & p[1] & g[0]);
    assign P = p[3] & p[2] & p[1] & p[0];


endmodule

module Add(
    input       [31:0]          a,
    input       [31:0]          b,
    output      [31:0]          sum,
);

    wire carry;

    carry_lookahead_adder_16bit adder (
        .a(a[15:0]),
        .b(b[15:0]),
        .cin(1'b0),
        .sum(sum[15:0]),
        .G(carry)
    );

    carry_lookahead_adder_16bit adder2 (
        .a(a[31:16]),
        .b(b[31:16]),
        .cin(carry),
        .sum(sum[31:16])
    );

endmodule