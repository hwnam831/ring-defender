/* 
 * Functions for reading and writing MSR registers, configuring CHA/CBO registers,
 * polling an address, and finding the slice counter (i.e., CBO/CHA) with highest number
 *
 * Copyright (c) 2019, Alireza Farshin, KTH Royal Institute of Technology - All Rights Reserved
 *
 *
 * The code for the functions rdmsr_on_cpu_0 and wrmsr_on_cpu_0 are
 * originally part of msr-tools.
 * The rest of the code has been inspired by Clémentine Maurice paper and her repository:
 * Paper:
 * 	Reverse Engineering Intel Last-Level Cache Complex Addressing Using Performance Counters
 * 	https://dl.acm.org/citation.cfm?id=2939211 doi>10.1007/978-3-319-26362-5_3
 * Repository:
 * 	https://github.com/clementine-m/msr-uncore-cbo
 */

#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <cpuid.h>
#include <inttypes.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h> /* for rdtscp and clflush */
#pragma optimize("gt",on)
#else
#include <x86intrin.h> /* for rdtscp and clflush */
#endif

/*
 * Definitions + MSR-related addresses
 */

/* Architecture related */
//#define SKYLAKE /* Can be changed to HASWELL or SKYLAKE -> Will be changed automatically by check_cpu.sh */
#define SKYLAKE_SERVER_MODEL 85
#define HASWELL_SERVER_MODEL 63

/* Number of polling for acquiring the slice */
#define NUMBER_POLLING 750

/* CBO/CHA addresses + values
 * For more info:
 * Check:
 * "Intel Xeon Processor E5 and E7 V3 Family Uncore Performance Monitoring"
 * Link: https://www.intel.com/content/www/us/en/processors/xeon/xeon-e5-v3-uncore-performance-monitoring.html
 * Or: 
 * "Intel® Xeon® Processor Scalable Family Uncore Reference Manual"
 * Link: https://www.intel.com/content/www/us/en/processors/xeon/scalable/xeon-scalable-uncore-performance-monitoring-manual.html
 *
 * Summary: 
 * For Setting up a Monitoring Session:
 * a)  Freeze all uncore counter -> Set U_MSR_PMON_GLOBAL_CTRL.frz_all
 * or
 * Freeze the box's counters -> Set Cn_MSR_PMON_BOX_CTL.frz
 *
 * b)  Enable counting for each monitor -> e.g., Set C0_MSR_PMON_CTL2.en
 *
 * c)  Select Event to monitor: program .ev_sel and .unmask_bits
 * e.g., Set C0_MSR_PMON_CTL2.{ev_sel,unmask} based on the table
 *
 * d)  Reset counters in each CHAx/CBox
 * e.g., For each CHA/CBo -> Set Cn_MSR_PMON_BOX_CTL[1:0] to 0x3
 *
 * e)  Select how to gather data -> Skip if polling, which is our case
 *
 * f)  Enable counting at the global level -> Set U_MSR_PMON_GLOBAL_CTRL.unfrz_all
 * or
 * Enable counting at the box level -> Set Cn_MSR_PMON_BOX_CTL.frz to 0
 */

/* MSR Addresses */
#define PMON_GLOBAL_CTL_ADDRESS 0xE01
const unsigned long long * CHA_CBO_EVENT_ADDRESS = \
	(unsigned long long []) {0x0700, 0x0710, 0x0720, 0x0730, 0x0740, 0x0750};
																

const unsigned long long * CHA_CBO_CTL_ADDRESS = \
	(unsigned long long []) {0x0706, 0x0716, 0x0726, 0x0736, 0x0746, 0x0756};

const unsigned long long * CHA_CBO_FILTER_ADDRESS = (unsigned long long []) {0x0E05, 0x0E15, 0x0E25, 0x0E35, 0x0E45, 0x0E55, 0x0E65, 0x0E75, 0x0E85, 0x0E95,
																0x0EA5, 0x0EB5, 0x0EC5, 0x0ED5, 0x0EE5, 0x0EF5, 0x0F05, 0x0F15, 0x0F25, 0x0F35, 0x0F45,
																0x0F55, 0x0F65, 0x0F75, 0x0F85, 0x0F95, 0x0FA5, 0x0FB5};

const unsigned long long * CHA_CBO_COUNTER_ADDRESS = \
	(unsigned long long []) {0x0706, 0x0716, 0x0726, 0x0736, 0x0746, 0x0756};

/* MSR Values */
#define ENABLE_COUNT_SKYLAKE 0x2000000000000000 
#define DISABLE_COUNT_SKYLAKE 0x8000000000000000
#define ENABLE_COUNT_SKYLAKES 0x20000000
#define DISABLE_COUNT_SKYLAKES 0x80000000
#define SELECTED_EVENT 0x448F34 /* Event: LLC_LOOKUP Mask: Any request (All snooping signals) */
#define RESET_COUNTERS 0x30002
#define FILTER_BOX_VALUE_SKYLAKE 0x01FE0000
#define FILTER_BOX_VALUE_SKYLAKES 0x007E0000


#ifdef SKYLAKE 
#define NUMBER_SLICES 24 /* Maximum number of slices in SkyLake architecture */
#define ENABLE_COUNT ENABLE_COUNT_SKYLAKE
#define DISABLE_COUNT DISABLE_COUNT_SKYLAKE	
#define FILTER_BOX_VALUE FILTER_BOX_VALUE_SKYLAKE															
#else
#define NUMBER_SLICES 6 /* Can be different for different CPUs */
#define ENABLE_COUNT ENABLE_COUNT_SKYLAKES
#define DISABLE_COUNT DISABLE_COUNT_SKYLAKES
#define FILTER_BOX_VALUE FILTER_BOX_VALUE_SKYLAKES	
#endif


/*
 * Read an MSR on CPU 0
 */

uint64_t rdmsr_on_cpu_0(uint32_t reg) {

	uint64_t data;
	int cpu = 0;
	char * msr_file_name = "/dev/cpu/0/msr";

	static int fd = -1;  

	if (fd < 0) {
		fd = open(msr_file_name, O_RDONLY);
		if(fd < 0) {
			if (errno == ENXIO) {
				fprintf(stderr, "rdmsr: No CPU %d\n", cpu);
				exit(2);
			} else if (errno == EIO) {
				fprintf(stderr, "rdmsr: CPU %d doesn't support MSRs\n",
					cpu);
				exit(3);
			} else {
				perror("rdmsr: open");
				exit(127);
			}
		}
	}
#ifdef DEBUG
		register uint64_t time1, time2;
		unsigned cycles_high, cycles_low, cycles_high1, cycles_low1;
	asm volatile ("CPUID\n\t"
				"RDTSC\n\t"
				"mov %%edx, %0\n\t"
				"mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "rax", "rbx", "rcx", "rdx");
#endif
	if (pread(fd, &data, sizeof data, reg) != sizeof data) {
		if (errno == EIO) {
			fprintf(stderr, "rdmsr: CPU %d cannot read "
				"MSR 0x%08"PRIx32"\n",
				cpu, reg);
			exit(4);
		} else {
			perror("rdmsr: pread");
			exit(127);
		}
	}
#ifdef DEBUG
		asm volatile ("RDTSCP\n\t"
				"mov %%edx, %0\n\t"
				"mov %%eax, %1\n\t"
				"CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "rax", "rbx", "rcx", "rdx");
			time1= (((uint64_t)cycles_high << 32) | cycles_low);
			time2= (((uint64_t)cycles_high1 << 32) | cycles_low1);
	printf("RDMSR took %d cycles\n",time2-time1);
#endif
  //close(fd);

	return data;
}

/*
 * Write to an MSR on CPU 0
 */

void wrmsr_on_cpu_0(uint32_t reg, int valcnt, uint64_t *regvals) {

	uint64_t data;
	char * msr_file_name = "/dev/cpu/0/msr";
	int cpu = 0;

	static int fd = -1;
	
	if(fd < 0){
		fd = open(msr_file_name, O_WRONLY);
		if (fd < 0) {
			if (errno == ENXIO) {
				fprintf(stderr, "wrmsr: No CPU %d\n", cpu);
				exit(2);
			} else if (errno == EIO) {
				fprintf(stderr, "wrmsr: CPU %d doesn't support MSRs\n",
					cpu);
				exit(3);
			} else {
				perror("wrmsr: open");
				exit(127);
			}
		}
	}

	while (valcnt--) {
#ifdef DEBUG
		register uint64_t time1, time2;
		unsigned cycles_high, cycles_low, cycles_high1, cycles_low1;
	asm volatile ("CPUID\n\t"
				"RDTSC\n\t"
				"mov %%edx, %0\n\t"
				"mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low):: "rax", "rbx", "rcx", "rdx");
#endif
		data=*regvals++;
		if (pwrite(fd, &data, sizeof data, reg) != sizeof data) {
			if (errno == EIO) {
				fprintf(stderr,
					"wrmsr: CPU %d cannot set MSR "
					"0x%08"PRIx32" to 0x%016"PRIx64"\n",
					cpu, reg, data);
				exit(4);
			} else {
				perror("wrmsr: pwrite");
				exit(127);
			}
		}
#ifdef DEBUG
		asm volatile ("RDTSCP\n\t"
				"mov %%edx, %0\n\t"
				"mov %%eax, %1\n\t"
				"CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1):: "rax", "rbx", "rcx", "rdx");
			time1= (((uint64_t)cycles_high << 32) | cycles_low);
			time2= (((uint64_t)cycles_high1 << 32) | cycles_low1);
	printf("WRMSR took %d cycles\n",time2-time1);
#endif
	}
	
  //close(fd);

	return;
}

/*
 * Polling one address
 */

void polling(void* address) {
	unsigned long i;
	//register int i asm ("eax");
	//register void* ptr asm ("ebx") = address;
	for (i=0;i<NUMBER_POLLING;i++) {
		//clflush(address);
		_mm_clflush(address);
	}
}


/*
 * Initialize uncore registers (CBo/CHA and Global MSR) before polling
 */

void uncore_init() {

	int i;

	/* Setup monitoring session */

	/* Disable counters */
	uint64_t register_value[] = {DISABLE_COUNT};
	register_value[0]=DISABLE_COUNT;
	wrmsr_on_cpu_0(PMON_GLOBAL_CTL_ADDRESS,1,register_value);

	/* Select the event to monitor */
	register_value[0]=SELECTED_EVENT;
	for(i=0; i<NUMBER_SLICES; i++) {
		wrmsr_on_cpu_0(CHA_CBO_EVENT_ADDRESS[i], 1, register_value);
	}

	//Reset Counters to Zero
	register_value[0]=0;
	for(i=0; i<NUMBER_SLICES; i++) {
		wrmsr_on_cpu_0(CHA_CBO_COUNTER_ADDRESS[i],1,register_value);
	}
	/* Reset CHA Counters
	register_value[0]=RESET_COUNTERS;
	for(i=0; i<NUMBER_SLICES; i++) {
		wrmsr_on_cpu_0(CHA_CBO_CTL_ADDRESS[i],1,register_value);
	}
	*/

	/* 
	//Set Filter BOX 
	register_value[0]=FILTER_BOX_VALUE;
	for(i=0; i<NUMBER_SLICES; i++) {
		wrmsr_on_cpu_0(CHA_CBO_FILTER_ADDRESS[i], 1, register_value);
	}*/

	/* Enable counting */
	register_value[0]=ENABLE_COUNT;
	wrmsr_on_cpu_0(PMON_GLOBAL_CTL_ADDRESS, 1, register_value);
}


/*
 * Read the CBo/CHA counters' value and find the one with maximum number
 */

int find_CHA_CBO() {

	int i;
	unsigned long long* CHA_CBO_value = calloc(NUMBER_SLICES, sizeof(unsigned long long));

	/* Read CHA/CBo counter's value */
	for(i=0; i<NUMBER_SLICES; i++){
		CHA_CBO_value[i] = rdmsr_on_cpu_0(CHA_CBO_COUNTER_ADDRESS[i]);
	}

	/* Find maximum */
	unsigned long long max_value=0;
	int max_index=0;
	for(i=0; i<NUMBER_SLICES; i++){
		//printf(" %llu\t%d\n", CHA_CBO_value[i],i);
		if(CHA_CBO_value[i]>max_value){
			max_value=CHA_CBO_value[i];
			max_index=i;
		}
	}
	return max_index;
}
