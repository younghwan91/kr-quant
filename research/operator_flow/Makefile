CFLAGS = -O3 -march=native -Wall
LDLIBS = -lm

all: sim sim2_oos

sim: sim.c
	gcc $(CFLAGS) -o sim sim.c $(LDLIBS)

sim2_oos: sim2_oos.c
	gcc $(CFLAGS) -o sim2_oos sim2_oos.c $(LDLIBS)

clean:
	rm -f sim sim2_oos

.PHONY: all clean
