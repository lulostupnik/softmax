#include "../include/softmax.h"

#define K_A 5  

//compile this impl in gem5
//see the asm code and see instructions used. 
//target some bottleneck by doing the instructions.
//do the algo with regular instructions + custom


static inline uint8_t bucket_round_KA(uint8_t delta) {
    return (uint8_t)((delta + (1u << (K_A - 1))) >> K_A); 
}


// Online denominator in Q0.7 + running max, pure shifts/adds.
static void denom_and_max_online_q7_optA(const int8_t *x, int n,
                                         uint16_t *sum_q7_out, int8_t *max_out)
{
    int8_t m = x[0];
    uint16_t s = ONE_Q7; // first term = 1.0

    for (int i = 1; i < n; ++i) {
        int8_t xi = x[i];
        if (xi <= m) {
            uint8_t d = (uint8_t)(m - xi);
            uint8_t bucket = bucket_round_KA(d);
            s += (ONE_Q7 >> bucket);
        } else {
            uint8_t d = (uint8_t)(xi - m);
            uint8_t bucket = bucket_round_KA(d);
            s = (s >> bucket) + ONE_Q7;
            m = xi;
        }
    }
    *sum_q7_out = s;
    *max_out    = m;
}

// Integer-only softmax using Option A; outputs Q0.7 [0..127].
void softmax_int8_optionA(const int8_t *x, int n, int8_t *y)
{
    if (n <= 0) return;

    int8_t m;
    uint16_t sum_q7;
    denom_and_max_online_q7_optA(x, n, &sum_q7, &m);

    // inv_q15 ≈ (127 / sum_q7) in Q15 (one 32-bit div per row)
    uint16_t inv_q15 = (uint16_t)(((uint32_t)127u << 15) / (sum_q7 ? sum_q7 : 1)); 

    for (int i = 0; i < n; ++i) {
        uint8_t d = (uint8_t)(m - x[i]);
        uint8_t bucket = bucket_round_KA(d);
        uint16_t num_q7 = (ONE_Q7 >> bucket);              // 2^{-bucket} in Q0.7
        uint32_t t = (uint32_t)inv_q15 * (uint32_t)num_q7; // Q15 * Q7 -> Q22
        uint16_t q = (uint16_t)((t + (1u<<14)) >> 15);     // -> Q7 with rounding
        if (q > 127) q = 127;
        y[i] = (int8_t)q;
    }
}

#include <stdio.h>

int main() {
    int8_t input[] = {10, 20, 30, 40, 50};
    int8_t output[5];
    int n = 5;

    printf("Input: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", input[i]);
    }
    printf("\n");

    softmax_int8_optionA(input, n, output);

    printf("Output: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", output[i]);
    }
    printf("\n");

    return 0;
}
