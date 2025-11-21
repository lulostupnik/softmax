#include "../include/softmax.h"

#define K_A 5

// compile this impl in gem5
// see the asm code and see instructions used.
// target some bottleneck by doing the instructions.
// do the algo with regular instructions + custom


// --- 1) Bucket rounding ---
// BOTTLENECK C OP: (delta + 16) >> 5
// Inlined in softmax_int8_optionA, e.g. around 0x1071e..0x10722:
//   addiw a5, a5, 16      ; delta + 16
//   srliw a5, a5, 0x5     ; >> 5 (K_A)
// Also used in the first loop (~0x106c2..0x106ca) for both branches.
static inline uint8_t bucket_round_KA(uint8_t delta) {
    return (uint8_t)((delta + (1u << (K_A - 1))) >> K_A);
}


// Online denominator in Q0.7 + running max, pure shifts/adds.
static void denom_and_max_online_q7_optA(const int8_t *x, int n,
                                         uint16_t *sum_q7_out, int8_t *max_out)
{
    int8_t m = x[0];
    uint16_t s = ONE_Q7; // first term = 1.0

    // BOTTLENECK: first pass over the vector (O(n)).
    // This whole loop is inlined into softmax_int8_optionA:
    //   Loop body roughly at 0x106a6..0x106e8.
    //
    // Key pieces seen in objdump:
    //   106a6: lb   t1,0(a7)          ; xi = x[i]
    //   106aa: zext.b t3,t4          ; t3 = (uint8_t)m
    //   106ae: zext.b a6,t1          ; a6 = (uint8_t)xi
    //
    //   106b2: subw  a4,t3,a6        ; d1 = m - xi
    //   106b6: subw  a5,a6,t3        ; d2 = xi - m
    //   106ba: zext.b a4,a4          ; |d1| in 0..255
    //   106be: zext.b a5,a5          ; |d2| in 0..255
    //
    //   106c2: addiw a4,a4,16        ; (d1 + 16) >> 5 -> bucket (xi <= m)
    //   106c4: addiw a5,a5,16        ; (d2 + 16) >> 5 -> bucket (xi > m)
    //   106c6: srliw a4,a4,0x5
    //   106ca: srliw a5,a5,0x5
    //
    //   106ce: sraw  a5,a3,a5        ; s >> bucket (candidate for xi > m path)
    //   106d2: sraw  a4,t6,a4        ; ONE_Q7 >> bucket
    //   106d6: addw  a3,a3,a4        ; s += (ONE_Q7 >> bucket)    (xi <= m)
    //   106d8: addi  a5,a5,128       ; (s >> bucket) + ONE_Q7     (xi > m)
    //
    //   106dc: bge   t4,t1,1069a     ; if (xi <= m) branch
    //   106e0: slli  a3,a5,0x30      ; else: s = (s >> bucket) + ONE_Q7
    //   106e6: srli  a3,a3,0x30
    //   106e8: mv    t4,t1           ; m = xi (update running max)
    for (int i = 1; i < n; ++i) {
        int8_t xi = x[i];
        if (xi <= m) {
            uint8_t d = (uint8_t)(m - xi);
            uint8_t bucket = bucket_round_KA(d);  // see addiw+srliw pattern above
            s += (ONE_Q7 >> bucket);
        } else {
            uint8_t d = (uint8_t)(xi - m);
            uint8_t bucket = bucket_round_KA(d);  // same bucket_round_KA asm sequence
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
    denom_and_max_online_q7_optA(x, n, &sum_q7, &m);  // inlined in objdump

    // BOTTLENECK: 32-bit integer division to compute inv_q15
    //
    // C:
    //   inv_q15 ≈ (127 / sum_q7) in Q15 (one 32-bit div per row)
    //   uint16_t inv_q15 = (uint16_t)(((uint32_t)127u << 15) /
    //                                 (sum_q7 ? sum_q7 : 1));
    //
    // Objdump (softmax_int8_optionA tail, around 0x106f2..0x10762):
    //   106f2: slliw a5,a5,0x10        ; zero-extend sum_q7 (a5)
    //   106f6: srliw a5,a5,0x10
    //   106fa: lui   a3,0x3f8          ; a3 = 127 << 15  (0x3f8000)
    //   106fe: bne   a5,a4,10762       ; if (sum_q7 != 1) go do division
    //   ...
    //   1075e: lui   a3,0x3f8          ; re-load numerator
    //   10762: divuw a3,a3,a5          ; (127 << 15) / sum_q7
    //   10766: slli  a3,a3,0x30        ; keep low 16 bits
    //   1076a: srli  a3,a3,0x30        ; a3 is uint16_t inv_q15
    uint16_t inv_q15 = (uint16_t)(((uint32_t)127u << 15) / (sum_q7 ? sum_q7 : 1));

    // BOTTLENECK: second pass over the vector (O(n)),
    //             mul + shift + clamp per element.
    //
    // For the loop body, objdump (around 0x10706..0x10740):
    //
    //   10706: add  a1,a1,a0          ; end_ptr = x + n (loop bound)
    //   10708: li   t1,128            ; t1 = ONE_Q7
    //   1070c: lui  a7,0x4            ; a7 = 1<<14 (rounding bias 16384)
    //   1070e: li   a4,127            ; saturation limit
    //
    //   Loop @0x10712:
    //   10712: lbu  a5,0(a0)          ; load x[i]
    //   10716: subw a5,a6,a5          ; d = m - x[i]
    //   1071a: zext.b a5,a5           ; cast to uint8_t
    //   1071e: addiw a5,a5,16         ; (d + 16)
    //   10722: srliw a5,a5,0x5        ; >> 5 -> bucket
    //   10726: sraw a5,t1,a5          ; num_q7 = ONE_Q7 >> bucket
    //   1072a: mulw a5,a5,a3          ; t = inv_q15 * num_q7 (Q15 * Q7)
    //   1072e: addw a5,a5,a7          ; + (1<<14) for rounding
    //   10732: srliw a5,a5,0xf        ; >> 15 -> Q7 (q)
    //   10736: bltu a4,a5,10746       ; if (q > 127) jump to saturation
    //   1073a: sb   a5,0(a2)          ; y[i] = q
    //   1073e: addi a0,a0,1           ; x++
    //   10740: addi a2,a2,1           ; y++
    //   10744: bne  a1,a0,10712       ; loop
    //
    //   Saturation path (q > 127) at 0x10746..):
    //   10746: sb   a4,0(a2)          ; y[i] = 127
    for (int i = 0; i < n; ++i) {
        uint8_t d = (uint8_t)(m - x[i]);
        uint8_t bucket = bucket_round_KA(d);          // addiw+srliw as above

        uint16_t num_q7 = (ONE_Q7 >> bucket);         // 2^{-bucket} in Q0.7
        // Objdump: sraw t1 >> bucket (t1 = 128) -> num_q7 in a5

        uint32_t t = (uint32_t)inv_q15 * (uint32_t)num_q7;
        // Objdump: mulw a5,a5,a3          ; inv_q15 (a3) * num_q7 (a5)

        uint16_t q = (uint16_t)((t + (1u<<14)) >> 15); // -> Q7 with rounding
        // Objdump:
        //   addw  a5,a5,a7   ; a7 = 1<<14
        //   srliw a5,a5,0xf  ; >> 15

        if (q > 127) q = 127;
        // Objdump:
        //   bltu  a4,a5,10746 ; if (q > 127) ...
        //   ...
        //   sb    a4,0(a2)    ; clamp to 127

        y[i] = (int8_t)q;
        // Objdump: sb a5,0(a2) or sb a4,0(a2) depending on clamp
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
