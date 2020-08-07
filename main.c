#include <gcrypt.h>
#define NEED_GCRYPT_VERSION "1.6.0"

int main(int argc, char **argv){
    if (!gcry_check_version(NEED_GCRYPT_VERSION)){
        fprintf (stderr, "libgcrypt is too old (need %s, have %s)\n",
         NEED_GCRYPT_VERSION, gcry_check_version (NULL));
        exit (2);
    }

    gcry_error_t err = 0;

    /* We don't want to see any warnings, e.g. because we have not yet
     parsed program options which might be used to suppress such
     warnings. */
    gcry_control (GCRYCTL_SUSPEND_SECMEM_WARN);

    /* Allocate a pool of 16k secure memory. */
    gcry_control (GCRYCTL_INIT_SECMEM, 16384, 0);

    /* It is now okay to let Libgcrypt complain when there was/is
        a problem with the secure memory. */
    gcry_control (GCRYCTL_RESUME_SECMEM_WARN);

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

    FILE *pvtf = fopen("pvtkey", "r");
    fread(buf, 2048, 1, pvtf);
    fclose(pvtf);

    gcry_sexp_build(&pvtkey, NULL, buf);

    /*
    gcry_sexp_sprint(pubkey, GCRYSEXP_FMT_ADVANCED, buf2, 2047);
    printf("%s\n",buf2);

    gcry_sexp_sprint(pvtkey, GCRYSEXP_FMT_ADVANCED, buf2, 2047);
    printf("%s\n",buf2);
    */

    const char* msg = "Hello World";
    


}