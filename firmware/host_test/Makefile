# Host build of the exact device frontend (reference FFT) for parity testing.
# Build from WSL:  make -C /mnt/c/Users/bornt/Desktop/Bird_Acoustics/firmware/host_test
CC ?= gcc
CFLAGS = -O2 -Wall -I../main -I../main/model

parity_main: parity_main.c ../main/frontend.c ../main/frontend_fft_ref.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f parity_main
