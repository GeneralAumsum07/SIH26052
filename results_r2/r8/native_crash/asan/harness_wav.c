/* Sweep harness: clean.wav + mix.wav (float32 WAV, channel 0 of mix = raw system), normalised as pesq/_pesq.py does.
   Built with the debug copy that flags a negative utterance start; prints one line per clip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pesq.h"
#include "pesqio.h"
#include "pesqmain.h"

extern int g_negstart;

static float *readwav(const char *p, int chan, long *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(2); }
    unsigned char h[12]; fread(h, 1, 12, f);
    int nch = 1; float *out = NULL;
    for (;;) {
        char id[4]; unsigned int sz; if (fread(id, 1, 4, f) != 4) break; fread(&sz, 4, 1, f);
        if (!memcmp(id, "fmt ", 4)) { unsigned char b[64]; fread(b, 1, sz, f); nch = b[2] | (b[3] << 8); }
        else if (!memcmp(id, "data", 4)) {
            float *all = malloc(sz); fread(all, 1, sz, f); long frames = sz / 4 / nch;
            out = malloc(frames * 4); for (long i = 0; i < frames; i++) out[i] = all[i * nch + chan];
            free(all); *n = frames; break;
        } else fseek(f, sz + (sz & 1), SEEK_CUR);
    }
    fclose(f); return out;
}

int main(int argc, char **argv) {
    long nr, nd; float *r = readwav(argv[1], 0, &nr), *d = readwav(argv[2], 0, &nd);
    float m = 0; for (long i = 0; i < nr; i++) { float a = r[i] < 0 ? -r[i] : r[i]; if (a > m) m = a; }
    for (long i = 0; i < nd; i++) { float a = d[i] < 0 ? -d[i] : d[i]; if (a > m) m = a; }
    for (long i = 0; i < nr; i++) r[i] = r[i] / m;
    for (long i = 0; i < nd; i++) d[i] = d[i] / m;
    long error_flag = 0; char *error_type = "unknown";
    select_rate(16000, &error_flag, &error_type);
    SIGNAL_INFO ref_info, deg_info; ERROR_INFO err_info;
    memset(&ref_info, 0, sizeof ref_info); memset(&deg_info, 0, sizeof deg_info); memset(&err_info, 0, sizeof err_info);
    ref_info.Nsamples = nr; ref_info.input_filter = 2; ref_info.data = r;
    deg_info.Nsamples = nd; deg_info.input_filter = 2; deg_info.data = d;
    err_info.mode = WB_MODE;
    pesq_measure(&ref_info, &deg_info, &err_info, &error_flag, &error_type);
    printf("%s flag %ld mos %.6f negstart %d\n", argv[3], error_flag, err_info.mapped_mos, g_negstart);
    return 0;
}
