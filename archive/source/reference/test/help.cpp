#include <stdio.h>

void helpmsg(char * exe)
{
    printf("This is a help message.\n");
    printf("usage:%s -m M -n N -k K -r repeat -c 0/1 -t threads\n", exe);
    printf("-m: M dimension length.\n");
    printf("-n: N dimension length.\n");
    printf("-k: K dimension length.\n");
    printf("-r: repeat times.\n");
    printf("-c: 0 not check correctness, 1 check correctness.\n");
    printf("-t: thread num.\n");
}