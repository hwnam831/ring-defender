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

    gcry_sexp_t rsa_parms;
    gcry_sexp_t rsa_keypair;

    err = gcry_sexp_build(&rsa_parms, NULL, "(genkey (rsa (nbits 4:1024)))");
    if (err) {
        fprintf(stderr, "gcrypt: failed to create rsa params");
        exit (2);
    }

    err = gcry_pk_genkey(&rsa_keypair, rsa_parms);
    if (err) {
        fprintf(stderr, "gcrypt: failed to create rsa key pair");
        exit (2);
    }

    char buf[2048];
    //gcry_sexp_t pubkey = gcry_sexp_nth(gcry_sexp_nth(rsa_keypair,1),1);
    gcry_sexp_t pubkey = gcry_sexp_find_token(rsa_keypair, "public-key", 0);
    //gcry_sexp_t pvtkey = gcry_sexp_nth(gcry_sexp_nth(rsa_keypair,2),1);
    gcry_sexp_t pvtkey = gcry_sexp_find_token(rsa_keypair, "private-key", 0);
    //printf("%d\n", gcry_sexp_length(pubkey));
    gcry_sexp_sprint(pubkey, GCRYSEXP_FMT_ADVANCED, buf, 2047);

    printf("%s\n",buf);
    /*
    gcry_sexp_t key2;
    err = gcry_sexp_build(&key2, NULL, buf);
    gcry_sexp_sprint(key2, GCRYSEXP_FMT_ADVANCED, buf, 2047);

    printf("%s\n",buf);
    */

    FILE *pubf = fopen("pubkey", "w");
    fputs(buf, pubf);
    fclose(pubf);

    gcry_sexp_sprint(pvtkey, GCRYSEXP_FMT_ADVANCED, buf, 2047);

    printf("%s\n",buf);

    FILE *pvtf = fopen("pvtkey", "w");
    fputs("(private-key\n(rsa\n", pvtf);
    gcry_sexp_sprint(gcry_sexp_find_token(pvtkey, "n", 0), GCRYSEXP_FMT_ADVANCED, buf, 2047);
    fputs(buf, pvtf);
    gcry_sexp_sprint(gcry_sexp_find_token(pvtkey, "e", 0), GCRYSEXP_FMT_ADVANCED, buf, 2047);
    fputs(buf, pvtf);
    gcry_sexp_sprint(gcry_sexp_find_token(pvtkey, "d", 0), GCRYSEXP_FMT_ADVANCED, buf, 2047);
    fputs(buf, pvtf);
    fputs(")\n)", pvtf);
    fclose(pvtf);

    


}