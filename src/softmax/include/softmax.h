#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>



#ifndef softmax_h
#define softmax_h

// ===================== Shared config =====================
#define B 8
#define LOG2_B 3
#define ONE_Q7 ((uint16_t)128u)   // 1.0 in Q0.7

// Uncomment to enable optional survivor clipping after quantization
// #define ITA_CLIP_SURVIVORS 1

// ===================== ITA epsilon quantizer =====================
#ifndef LOG2E
#define LOG2E 1.4426950408889634074  // log2(e)
#endif


void softmax_int8_optionA(const int8_t *x, int n, int8_t *y);
void softmax_int8_optionB(const int8_t *x, int n, int8_t *y);



#endif // softmax_h