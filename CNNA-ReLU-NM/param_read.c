#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "param_read.h"

float * read_params(const char * path, int entries_count)
{
    FILE * stream;
    float * params;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\nExiting\n", path);
        exit(0);
        return NULL;
    }

    params = malloc(entries_count * sizeof(float));

    if (params == NULL) {
        fprintf(stderr, "Could not allocated memory for %d params\n", entries_count);
        fclose(stream);
        return NULL;
    }

    char line[256];
    int count = 0;
    float value = 0.0f;

    while(fgets(line, sizeof(line), stream))
    {
        // printf("%d\n", count);
        value = atof(line);
        // printf("%f\n", value);
        *(params+count) = value;        
        // printf("%f\n", *(params+count));
        count = count + 1;
    }

    // if (*entries_count != fread(params, sizeof(float), *entries_count, stream)) {
    //     fprintf(stderr, "Could not read %d params from: %s\n", *entries_count, path);
    //     free(params);
    //     fclose(stream);
    //     return NULL;
    // }

    fclose(stream);

    return params;
}