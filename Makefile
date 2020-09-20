CC = gcc
GCRYPTFLAGS = -lgcrypt -lrt -lpthread
CFLAGS = -Wl,-rpath,/home/hwnam/gcrypt/lib -g -I/home/hwnam/gcrypt/include -L/home/hwnam/gcrypt/lib 

EXECS = keygen victim attacker

default: all

%.o: %.c
	$(CC) -c $(CFLAGS) -o $*.o $*.c $(GCRYPTFLAGS)

keygen: keygen.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

victim: main.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)

attacker: L3_access_measurement.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)
all: $(EXECS)

run: victim
	sudo ./victim > victim.log

clean:
	rm $(EXECS) *.o *.log
