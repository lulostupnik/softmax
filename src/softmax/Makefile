# Makefile for Integer-Only Softmax
# Portable C99 with strict warnings

CC = gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-parameter -std=c99 -pedantic
LDFLAGS = -lm

# Directories
SRCDIR = src
INCDIR = include
TESTDIR = tests
BUILDDIR = build

# Source files
SOURCES = $(SRCDIR)/softmax_int8.c \
          $(SRCDIR)/lut_exp2.c \
          $(SRCDIR)/quant_utils.c

# Object files
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)

# Test executable
TEST_SRC = $(TESTDIR)/test_softmax.c
TEST_BIN = $(BUILDDIR)/test_softmax

# Include paths
INCLUDES = -I$(INCDIR)

# Targets
.PHONY: all clean test run help

all: $(TEST_BIN)
	@echo "✓ Build complete"

# Create build directory
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# Compile object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Link test executable
$(TEST_BIN): $(OBJECTS) $(TEST_SRC) | $(BUILDDIR)
	@echo "Linking $@..."
	$(CC) $(CFLAGS) $(INCLUDES) $(TEST_SRC) $(OBJECTS) $(LDFLAGS) -o $@

# Run tests
test: $(TEST_BIN)
	@echo "Running tests..."
	@./$(TEST_BIN)

run: test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILDDIR)
	@echo "✓ Clean complete"

# Help target
help:
	@echo "Integer-Only Softmax - Makefile Targets"
	@echo "========================================"
	@echo "  make all    - Build test executable"
	@echo "  make test   - Build and run tests"
	@echo "  make run    - Alias for 'make test'"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"
	@echo ""
	@echo "Compiler: $(CC)"
	@echo "Flags:    $(CFLAGS)"
