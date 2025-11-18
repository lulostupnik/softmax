#include "softmax.h"

// K_B defines the granularity of our exponential approximation
// We split delta values into coarse buckets (step=32) and fine fractions
#define K_B 5  // coarse buckets: step=32

// Lookup table for 2^{-(f/32)} in Q15 fixed-point format for f = 0..31
// Q15 means 15 fractional bits: value = (integer / 32768)
// This approximates exp(-x) ≈ 2^(-x/ln(2)) for small increments
static const uint16_t LUT_Q15_2P_MINUS_F_OVER_32[32] = {
    32768, 32066, 31379, 30706, 30048, 29405, 28774, 28158,
    27554, 26964, 26386, 25821, 25268, 24726, 24196, 23678,
    23170, 22674, 22188, 21713, 21247, 20792, 20347, 19911,
    19484, 19066, 18658, 18258, 17867, 17484, 17109, 16743
};

// Extract coarse bucket: how many full steps of 32
static inline uint8_t bucket_KB(uint8_t delta) { return (uint8_t)(delta >> K_B); }

// Extract fine fraction: remainder within the bucket (0-31)
static inline uint8_t frac_KB(uint8_t delta)   { return (uint8_t)(delta & ((1u<<K_B)-1)); }


/**
 * Compute denominator (sum of exp(x[i] - max)) and max value in one pass.
 * Uses online algorithm to avoid overflow by tracking running maximum.
 * 
 * @param x Input array in Q0.7 fixed-point (range: -128 to 127)
 * @param n Number of elements
 * @param sum_q7_out Output: denominator sum in Q0.7 format
 * @param max_out Output: maximum value from input array
 */
static void denom_and_max_online_q7_optB(const int8_t *x, int n,
                                         uint16_t *sum_q7_out, int8_t *max_out)
{
    // Initialize with first element
    int8_t m = x[0];
    uint16_t s = ONE_Q7; // first term = exp(x[0] - x[0]) = 1.0 in Q0.7

    for (int i = 1; i < n; ++i) {
        int8_t xi = x[i];
    
        if (xi <= m) {
            // Current element is not the new max
            // Compute exp(xi - m) and add to sum
            uint8_t d = (uint8_t)(m - xi);        // delta = max - xi
            uint8_t b = bucket_KB(d);             // coarse bucket (d / 32)
            uint8_t f = frac_KB(d);               // fine fraction (d % 32)
            uint16_t lut = LUT_Q15_2P_MINUS_F_OVER_32[f]; // lookup 2^(-f/32)
            
            // Compute 2^(-d) ≈ 2^(-b) * 2^(-f/32)
            uint32_t term = ((uint32_t)(ONE_Q7 >> b) * (uint32_t)lut);
            term = (term + (1u<<14)) >> 15; // Round and convert Q15 back to Q0.7
            s += (uint16_t)term;
        } else {
            // New maximum found. Need to rescale all previous terms
            uint8_t d = (uint8_t)(xi - m);        // delta = new_max - old_max
            uint8_t b = bucket_KB(d);
            uint8_t f = frac_KB(d);
            uint16_t lut = LUT_Q15_2P_MINUS_F_OVER_32[f];
            
            // Rescale existing sum: s_new = s_old * exp(old_max - new_max)
            uint32_t scaled = ((uint32_t)(s >> b) * (uint32_t)lut);
            s = (uint16_t)((scaled + (1u<<14)) >> 15); // rescale old sum in Q0.7
            s += ONE_Q7; // add new dominant term exp(xi - xi) = 1.0
            m = xi;      // update maximum
        }
    }
    *sum_q7_out = s;
    *max_out    = m;
}

/**
 * Compute softmax using integer arithmetic with lookup table optimization.
 * softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
 *               = exp(x[i] - max) / sum(exp(x[j] - max))
 * 
 * @param x Input array in Q0.7 fixed-point format
 * @param n Number of elements
 * @param y Output array in Q0.7 fixed-point format (range: 0 to 127)
 */
void softmax_int8_optionB(const int8_t *x, int n, int8_t *y)
{
    if (n <= 0) return;

    // Step 1: Compute denominator sum and max value
    int8_t m;
    uint16_t sum_q7;
    denom_and_max_online_q7_optB(x, n, &sum_q7, &m);

    // Step 2: Compute reciprocal of sum (one division per row)
    // inv_q15 ≈ (127 / sum_q7) in Q15 format for subsequent multiplications
    uint16_t inv_q15 = (uint16_t)(((uint32_t)127u << 15) / (sum_q7 ? sum_q7 : 1));

    // Step 3: Compute each output element: y[i] = exp(x[i] - max) / sum
    for (int i = 0; i < n; ++i) {
        // Compute exp(x[i] - max) using bucket + LUT approach
        uint8_t d = (uint8_t)(m - x[i]);          // delta from max
        uint8_t b = bucket_KB(d);                  // coarse bucket
        uint8_t f = frac_KB(d);                    // fine fraction
        uint16_t lut = LUT_Q15_2P_MINUS_F_OVER_32[f];

        // Compute numerator: exp(x[i] - max) in Q0.7
        uint32_t num = ((uint32_t)(ONE_Q7 >> b) * (uint32_t)lut); // Q7*Q15
        uint16_t num_q7 = (uint16_t)((num + (1u<<14)) >> 15);     // -> Q0.7

        // Divide by sum using precomputed reciprocal
        uint32_t t = (uint32_t)inv_q15 * (uint32_t)num_q7;        // Q15*Q7
        uint16_t q = (uint16_t)((t + (1u<<14)) >> 15);            // -> Q0.7
        
        // Clamp to valid range [0, 127]
        if (q > 127) q = 127;
        y[i] = (int8_t)q;
    }
}

