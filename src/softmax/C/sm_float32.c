#include <stdio.h>
#include <math.h>

/*
 * Standard float32 Softmax
 * Input:  float x[n]
 * Output: float y[n] where sum(y) = 1
 * Stable for large values (subtracts max)
 */

void softmax_f32(const float *input, float *output, int length) {
    if (length <= 0) return;

    // 1. Find max value (for numerical stability)
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }

    // 2. Compute exp(x_i - max_val) and accumulate sum
    float sum_exp = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }

    // 3. Normalize so the sum of outputs = 1
    for (int i = 0; i < length; i++) {
        output[i] /= sum_exp;
    }
}

int main(void) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int n = sizeof(input) / sizeof(input[0]);
    float output[n];

    softmax_f32(input, output, n);

    printf("Input:  ");
    for (int i = 0; i < n; i++) printf("%.2f ", input[i]);
    printf("\nOutput (softmax): ");
    for (int i = 0; i < n; i++) printf("%.6f ", output[i]);
    printf("\n");

    return 0;
}
