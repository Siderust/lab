# Build the libnova adapter
# Compiles required libnova sources from the vendored submodule and links
# the adapter against them.

LIBNOVA_SRC_DIR := ../../../libnova/src

# Only the source files we actually need (precession, nutation, sidereal time
# and their transitive dependencies: utility, dynamical_time).
LIBNOVA_SRCS := \
	$(LIBNOVA_SRC_DIR)/precession.c \
	$(LIBNOVA_SRC_DIR)/nutation.c \
	$(LIBNOVA_SRC_DIR)/sidereal_time.c \
	$(LIBNOVA_SRC_DIR)/utility.c \
	$(LIBNOVA_SRC_DIR)/dynamical_time.c

LIBNOVA_OBJS := $(patsubst $(LIBNOVA_SRC_DIR)/%.c, build/libnova/%.o, $(LIBNOVA_SRCS))

CC       := gcc
CFLAGS   := -O2 -Wall -std=c11 -D_GNU_SOURCE -I$(LIBNOVA_SRC_DIR) \
             -Wno-unused-function -Wno-unused-variable -Wno-sign-compare \
             -Wno-misleading-indentation -Wno-implicit-function-declaration \
             -Wno-int-conversion
LDFLAGS  := -lm

TARGET   := build/libnova_adapter

.PHONY: all clean

all: $(TARGET)

$(TARGET): build/main.o $(LIBNOVA_OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

build/main.o: main.c | build
	$(CC) $(CFLAGS) -c -o $@ $<

build/libnova/%.o: $(LIBNOVA_SRC_DIR)/%.c | build/libnova
	$(CC) $(CFLAGS) -c -o $@ $<

build:
	mkdir -p build

build/libnova:
	mkdir -p build/libnova

clean:
	rm -rf build
