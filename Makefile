CC = gcc
GCRYPTFLAGS = -lgcrypt 
CFLAGS = -g -I/home/hwnam/gcrypt/include -L/home/hwnam/gcrypt/lib 

EXECS = keygen victim

default: all

%.o: %.c
	$(CC) -c $(CFLAGS) -o $*.o $*.c $(GCRYPTFLAGS)

keygen: keygen.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

victim: main.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

all: $(EXECS)

clean:
	rm $(EXECS) *.o
