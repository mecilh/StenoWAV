#ifndef MATHSTENO_H
#define MATHSTENO_H

#include <complex>
#include <iostream>
#include <cmath>
#include <vector>
#include <kfr/all.hpp>

using std::vector;

class MathSteno {

    public:
        class TimeDomain{
    private:
        static float rms_window(const vector<int16_t>& samples, int center, int window_size) {
            int half = window_size / 2;
            int start = std::max(0, center - half);
            int end = std::min((int)samples.size(), center + half);
            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                float s = (float)samples[i];
                sum += s * s;
            }
            return std::sqrt(sum / (end - start));
        }

        static float rms_window(const vector<uint8_t>& samples, int center, int window_size) {
            int half = window_size / 2;
            int start = std::max(0, center - half);
            int end = std::min((int)samples.size(), center + half);
            float sum = 0.0f;
            for (int i = start; i < end; i++) {
                float s = (float)samples[i] - 128.0f;
                sum += s * s;
            }
            return std::sqrt(sum / (end - start));
        }

    public:
        static float average_rms(const vector<int16_t>& samples) {
            float sum = 0.0f;
            for (auto s : samples) {
                float f = (float)s;
                sum += f * f;
            }
            return std::sqrt(sum / samples.size());
        }

        static float average_rms(const vector<uint8_t>& samples) {
            float sum = 0.0f;
            for (auto s : samples) {
                float f = (float)s - 128.0f;
                sum += f * f;
            }
            return std::sqrt(sum / samples.size());
        }

        static float rms_to_db(float rms, float reference) {
            if (rms < 1e-10f) return -100.0f;
            return 20.0f * std::log10(rms / reference);
        }

        static vector<int> validate_position_time_domain(const vector<int16_t>& samples, int window_size = 1024, float threshold_db = -20.0f) {
            vector<int> valid;
            float reference = 32768.0f;

            for (int i = 0; i < (int)samples.size(); i++) {
                float rms = rms_window(samples, i, window_size);
                float db = rms_to_db(rms, reference);
                if (db > threshold_db)
                    valid.push_back(i);
            }
            return valid;
        }

        static vector<int> validate_position_time_domain(const vector<uint8_t>& samples, int window_size = 1024, float threshold_db = -20.0f) {
            vector<int> valid;
            float reference = 128.0f;

            for (int i = 0; i < (int)samples.size(); i++) {
                float rms = rms_window(samples, i, window_size);
                float db = rms_to_db(rms, reference);
                if (db > threshold_db)
                    valid.push_back(i);
            }
            return valid;
        }
    };
};

#endif