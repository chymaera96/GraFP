import numpy as np
import json
import librosa

def calculate_t60(impulse_response, sr):
    # Convert to mono
    if impulse_response.ndim > 1:
        impulse_response = librosa.to_mono(impulse_response)

    # Calculate the squared magnitude of the impulse response
    magnitude_squared = np.square(impulse_response)

    # Calculate the cumulative sum of the squared magnitude, normalized to [0, 1]
    energy_decay = np.cumsum(magnitude_squared[::-1])[::-1]
    energy_decay /= energy_decay[0]

    # Convert to dB
    energy_decay_db = librosa.amplitude_to_db(energy_decay, ref=np.max)

    # Find the time index where the energy decays by 60 dB
    t60_index = np.where(energy_decay_db <= -60)[0][0]

    # Convert the index to time
    t60 = t60_index / sr

    return t60


def main():
    ir_file = '../datasets/ir/ir.json'
    import json

    with open(ir_file, 'r') as fp:
        ir_idx = json.load(fp)

    t60 = []
    for path in ir_idx['test']:
        y, sr = librosa.load(path, sr=16000)
        t60.append(calculate_t60(y, sr=16000))

    # Save path of t60 in range 0.4 to 0.6
    t60 = np.array(t60)
    idx = np.where((t60 >= 0.4) & (t60 <= 0.6))[0]
    paths = [ir_idx['test'][i] for i in idx]
    print(paths)
    print(len(paths))

if __name__ == '__main__':
    main()

