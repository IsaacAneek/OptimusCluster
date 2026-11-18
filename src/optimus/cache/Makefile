CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wno-deprecated-declarations -D_FILE_OFFSET_BITS=64
FUSE_CFLAGS := $(shell pkg-config --cflags fuse3)
FUSE_LIBS := $(shell pkg-config --libs fuse3)
LIBS := $(FUSE_LIBS) -lsqlite3 -lssl -lcrypto -lpthread

TARGET = fuse_cache

all: $(TARGET)

$(TARGET): fuse_cache.c
	$(CC) $(CFLAGS) $(FUSE_CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
