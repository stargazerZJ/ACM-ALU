.PHONY: all clean ripple carry

# Targets
all: ripple carry

# Testing Ripple-Carry Adder
ripple:
	@echo 'Now testing: Ripple-Carry Adder'
	@cat adder-ripple.v > adder.v
	@iverilog.exe -o ripple_adder_test.out test_adder.v || { echo 'Build failed for Ripple-Carry Adder'; exit 1; }
	@vvp.exe ripple_adder_test.out
	@rm -f ripple_adder_test.out adder.v
	@echo 'Ripple-Carry Adder test complete\n'

# Testing Carry-lookahead Adder
carry:
	@echo 'Now testing: Carry-lookahead Adder'
	@cat adder-carry.v > adder.v
	@iverilog.exe -o carry_adder_test.out test_adder.v || { echo 'Build failed for Carry-lookahead Adder'; exit 1; }
	@vvp.exe carry_adder_test.out
	@rm -f carry_adder_test.out adder.v
	@echo 'Carry-lookahead Adder test complete\n'

# Clean up generated files
clean:
	@rm -f ripple_adder_test carry_adder_test adder.v
	@echo 'Cleanup complete'