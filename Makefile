CC = gcc
GCRYPTFLAGS = -lgcrypt
CFLAGS = -g
EXECS = keygen defender


%.o: %.c
	$(CC) -c $(CFLAGS) -o $*.o $*.c $(GCRYPTFLAGS)

keygen: keygen.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

defender: main.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

all: $(EXECS)

clean:
	rm $(EXECS) *.o