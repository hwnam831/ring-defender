CC = gcc
GCRYPTFLAGS = -lgcrypt -lrt -lpthread
CFLAGS = -Wl,-rpath,/home/hwnam/gcrypt/lib -g -I/home/hwnam/gcrypt/include -L/home/hwnam/gcrypt/lib 
GCRYPTPATH = ~/libgcrypt
EXECS = keygen victim attacker

default: all

%.o: %.c
	$(CC) -c $(CFLAGS) -o $*.o $*.c $(GCRYPTFLAGS)

keygen: keygen.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

victim: main.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

attacker: L3_access_measurement.o lib/memory-utils.c lib/cache-utils.c lib/msr-utils.c
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

gcrypt: mpi-pow.c lib/memory-utils.c lib/cache-utils.c lib/msr-utils.c
	cp mpi-pow.c $(GCRYPTPATH)/mpi/
	cd $(GCRYPTPATH) && make && make install

all: $(EXECS)

run: victim
	sudo ./victim

clean:
	rm $(EXECS) *.o *.log
