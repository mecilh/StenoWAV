import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import random as rd
import struct


FFT_SIZE = 1024
HOP_SIZE = FFT_SIZE // 8


def apply_randomness(fft_data, k_factor, pre_places_positions):
    for index in range(int(len(fft_data) // k_factor)):
        if index not in pre_places_positions:
            random_value_r = rd.uniform(-5, 5)
            fft_data[int(index * k_factor) - 1] += random_value_r
    return fft_data


def apply_data(fft_data, k_factor, floats, positions):
    for index, pos in enumerate(positions):
        if index < len(floats):
            fft_data[int(pos * k_factor)] += floats[index]
    return fft_data


def process_fft(fft_data, k_factor):

    data_compress = compress("Hello World!")
    positions = []
    for index in range(0, len(data_compress)):
        prob_position = rd.randint(0,int(len(fft_data) // k_factor))
        if prob_position not in positions:
            positions.append(prob_position)
    apply_randomness(fft_data, k_factor, positions)
    apply_data(fft_data,k_factor, data_compress, positions)
    return fft_data


def overlap_add_process(audio, fft_size, hop_size, k_factor):
    window = np.hanning(fft_size)
    output = np.zeros(len(audio) + fft_size)

    norm = np.zeros(len(audio) + fft_size)

    for start in range(0, len(audio) - fft_size, hop_size):
        segment = audio[start:start + fft_size] * window
        fft_data = np.fft.rfft(segment)
        fft_data = process_fft(fft_data, k_factor)
        processed = np.fft.irfft(fft_data)
        output[start:start + fft_size] += processed * window
        norm[start:start + fft_size] += window ** 2
    norm = np.maximum(norm, 1e-8)  
    output = output / norm

    return output[:len(audio)]


def main():
    sample_rate, audio = wav.read("audio_sources/MyOwnSummer.wav")
    audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max

    k_factor = len(audio) / sample_rate
    left = overlap_add_process(audio[:, 0], FFT_SIZE, HOP_SIZE, k_factor)
    right = overlap_add_process(audio[:, 1], FFT_SIZE, HOP_SIZE, k_factor)
    output = np.column_stack((left, right))
    output = np.clip(output, -1 ,1)
    print("Operation completed")
    wav.write("audio_outputs/output.wav", sample_rate, output)
    print("Audio written")
    sd.play(output, sample_rate)
    sd.wait()



def compress(string):
    """
    Build IEEE 754 floats manually:
    - Sign: 1 bit (0 = positive)
    - Exponent: 8 bits (random 0-4)
    - Mantissa: 23 bits (7 random + 16 from 2 ASCII chars)
    """
    floats = []

    # Process 2 chars at a time
    for i in range(0, len(string), 2):
        chars = string[i:i+2].ljust(2, '\x00')  # Pad if odd length

        sign = 0  # Positive
        exponent = rd.randint(80, 125)  # Away from eps, not audible

        # Mantissa: 7 random bits + 16 bits from 2 chars
        random_7bits = rd.randint(0, 0x7F)  # 7 bits
        char_16bits = (ord(chars[0]) << 8) | ord(chars[1])  # 16 bits
        mantissa = (random_7bits << 16) | char_16bits  # 23 bits total

        # Assemble: sign(1) | exponent(8) | mantissa(23)
        float_bits = (sign << 31) | (exponent << 23) | mantissa
        float_val = struct.unpack('f', struct.pack('I', float_bits))[0]
        floats.append(float_val)

    return floats

    


if (__name__ == "__main__"):
    main()
