#ifndef MATHSTENO_H
#define MATHSTENO_H

#include <cmath>
#include <cstdint>
#include <vector>

using std::vector;

class MathSteno {
public:
    class TimeDomain {
    public:
        static float rms_to_db(float rms, float reference) {
            if (rms < 1e-10f) return -100.0f;
            return 20.0f * std::log10(rms / reference);
        }

        // Sliding-window RMS validation: O(N) instead of O(N*W).
        // Returns indices where the local RMS exceeds threshold_db.
        static vector<int> validate_position_time_domain(
            const vector<int16_t>& samples,
            int window_size = 1024,
            float threshold_db = -20.0f)
        {
            vector<int> valid;
            float reference = 32768.0f;
            int n = static_cast<int>(samples.size());
            if (n == 0) return valid;

            int half = window_size / 2;
            double running_sum = 0.0;
            int win_start = 0;
            int win_end = 0;

            // Initialize window for index 0
            int s0 = std::max(0, 0 - half);
            int e0 = std::min(n, 0 + half);
            for (int j = s0; j < e0; j++) {
                double s = static_cast<double>(samples[j]);
                running_sum += s * s;
            }
            win_start = s0;
            win_end = e0;

            for (int i = 0; i < n; i++) {
                int new_start = std::max(0, i - half);
                int new_end = std::min(n, i + half);

                // Remove samples that fell out of the window on the left
                for (int j = win_start; j < new_start; j++) {
                    double s = static_cast<double>(samples[j]);
                    running_sum -= s * s;
                }
                // Add samples that entered the window on the right
                for (int j = win_end; j < new_end; j++) {
                    double s = static_cast<double>(samples[j]);
                    running_sum += s * s;
                }

                win_start = new_start;
                win_end = new_end;

                int count = win_end - win_start;
                float rms = std::sqrt(static_cast<float>(running_sum / count));
                float db = rms_to_db(rms, reference);
                if (db > threshold_db) {
                    valid.push_back(i);
                }
            }
            return valid;
        }

        static vector<int> validate_position_time_domain(
            const vector<uint8_t>& samples,
            int window_size = 1024,
            float threshold_db = -20.0f)
        {
            vector<int> valid;
            float reference = 128.0f;
            int n = static_cast<int>(samples.size());
            if (n == 0) return valid;

            int half = window_size / 2;
            double running_sum = 0.0;
            int win_start = 0;
            int win_end = 0;

            int s0 = std::max(0, 0 - half);
            int e0 = std::min(n, 0 + half);
            for (int j = s0; j < e0; j++) {
                double s = static_cast<double>(samples[j]) - 128.0;
                running_sum += s * s;
            }
            win_start = s0;
            win_end = e0;

            for (int i = 0; i < n; i++) {
                int new_start = std::max(0, i - half);
                int new_end = std::min(n, i + half);

                for (int j = win_start; j < new_start; j++) {
                    double s = static_cast<double>(samples[j]) - 128.0;
                    running_sum -= s * s;
                }
                for (int j = win_end; j < new_end; j++) {
                    double s = static_cast<double>(samples[j]) - 128.0;
                    running_sum += s * s;
                }

                win_start = new_start;
                win_end = new_end;

                int count = win_end - win_start;
                float rms = std::sqrt(static_cast<float>(running_sum / count));
                float db = rms_to_db(rms, reference);
                if (db > threshold_db) {
                    valid.push_back(i);
                }
            }
            return valid;
        }

        // Single-sample RMS in a window (for bit-budget calculation during scatter).
        static float rms_window(const vector<int16_t>& samples, int center, int window_size) {
            int half = window_size / 2;
            int start = std::max(0, center - half);
            int end = std::min(static_cast<int>(samples.size()), center + half);
            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                float s = static_cast<float>(samples[i]);
                sum += s * s;
            }
            return std::sqrt(sum / (end - start));
        }

        static float rms_window(const vector<uint8_t>& samples, int center, int window_size) {
            int half = window_size / 2;
            int start = std::max(0, center - half);
            int end = std::min(static_cast<int>(samples.size()), center + half);
            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                float s = static_cast<float>(samples[i]) - 128.0f;
                sum += s * s;
            }
            return std::sqrt(sum / (end - start));
        }
    };
};

#endif
