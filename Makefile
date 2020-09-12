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

attacker: attacker.o
	$(CC) $(CFLAGS) -o $@ $< $(GCRYPTFLAGS)
all: $(EXECS)

run: victim
	sudo ./victim 1 3 > victim.log
	python3 parse.py

clean:
	rm $(EXECS) *.o *.log
