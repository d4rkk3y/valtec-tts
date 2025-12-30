# Valtec Vietnamese TTS

A Vietnamese Text-to-Speech system supporting multiple speakers with high-quality voice synthesis.

## Features

- Multi-speaker Vietnamese TTS
- **⚡ Ultra-fast inference** with GPU acceleration (RTF as low as **0.014**)
- Advanced Vietnamese text normalization and phonemization
- Natural prosody and intonation
- Auto-download pretrained models from Hugging Face
- Simple 2-line API for quick usage

## 🎧 Audio Examples

Listen to sample outputs from our TTS system:

| Speaker | Region | Gender | Sample Text | Audio |
|---------|--------|--------|-------------|-------|
| **NF** | Northern (Bắc) | Female | "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ." | [▶️ example_NF.wav](examples/example_NF.wav) |
| **SF** | Southern (Nam) | Female | "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ." | [▶️ example_SF.wav](examples/example_SF.wav) |
| **NM1** | Northern (Bắc) | Male | "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ." | [▶️ example_NM1.wav](examples/example_NM1.wav) |
| **SM** | Southern (Nam) | Male | "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ." | [▶️ example_SM.wav](examples/example_SM.wav) |
| **NM2** | Northern (Bắc) | Male | "Tiếng xe cộ nhộn nhịp và ánh nắng len qua từng con phố nhỏ." | [▶️ example_NM2.wav](examples/example_NM2.wav) |

> 💡 **Tip**: Clone the repository and listen to the files in the `examples/` folder for the best audio quality.

## 🎬 Live Demos

### Web Demo - Browser-based TTS

**[▶️ Watch Web Demo Video](https://github.com/tronghieuit/valtec-tts/raw/dev/examples/ValtecTTS%20-%20WEB.mp4)**

**Features:**
- ✅ Runs entirely in browser using ONNX Runtime Web
- ✅ No backend server required
- ✅ All 5 Vietnamese voices
- ✅ Real-time synthesis (~165MB models)

[See full documentation](deployments/web/README.md)

---

### Android Demo - On-Device TTS

**[▶️ Watch Android Demo Video](https://github.com/tronghieuit/valtec-tts/raw/dev/examples/ValtecTTS%20-%20Android.mp4)**

**Features:**
- ✅ On-device ONNX inference
- ✅ Offline-capable after installation
- ✅ Tested on Xiaomi 12S Pro (Snapdragon 8+ Gen 1)
- ✅ ~200-300ms inference time

[See full documentation](deployments/android/README.md)

## ⚡ Performance Benchmark

### Model Specifications

| Metric | Value |
|--------|-------|
| **Total Parameters** | **57.97M** (57,972,657) |
| Trainable Parameters | 57.97M |
| Sample Rate | 24,000 Hz |
| Architecture | VITS-based |

### Inference Speed

Benchmark conducted on:
- **CPU**: Intel Core i5-14500 (14 cores)
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB VRAM, Compute Capability 8.9)
- **PyTorch**: 2.9.1+cu128
- **OS**: Linux

#### CPU Inference (Intel i5-14500)

| Input Length | Inference Time | Audio Length | RTF |
|-------------|----------------|--------------|-----|
| Short (15 chars) | 391.59ms ± 268.76ms | 0.73s | 0.5354 ✅ |
| Medium (52 chars) | 853.95ms ± 262.50ms | 2.23s | 0.3831 ✅ |
| Long (120 chars) | 1653.24ms ± 384.59ms | 4.91s | 0.3366 ✅ |

#### 🚀 CUDA Inference (RTX 4060 Ti)

| Input Length | Inference Time | Audio Length | RTF | Speedup vs CPU |
|-------------|----------------|--------------|-----|----------------|
| Short (15 chars) | **45.14ms** ± 24.69ms | 0.73s | **0.0617** ✅ | **8.7x** |
| Medium (52 chars) | **51.82ms** ± 29.35ms | 2.23s | **0.0232** ✅ | **16.5x** |
| Long (120 chars) | **68.88ms** ± 28.14ms | 4.91s | **0.0140** ✅ | **24.0x** |

> **RTF (Real-Time Factor)**: Ratio of processing time to audio duration. RTF < 1 means faster than real-time.
> 
> 🔥 **With CUDA, the model generates audio 70x faster than real-time for long texts!**

### GPU Speedup Summary

```
⚡ GPU Speedup over CPU:
   Short text:  8.7x faster
   Medium text: 16.5x faster  
   Long text:   24.0x faster
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- **Linux recommended** for best phonemization quality (viphoneme)

> **Note:** On Windows, the system uses a fallback phonemizer. For production-quality Vietnamese pronunciation, please run on Linux or WSL.

## Installation

### From Git

```bash
pip install git+https://github.com/tronghieuit/valtec-tts.git
```

### From Source

```bash
git clone https://github.com/tronghieuit/valtec-tts.git
cd valtec-tts
pip install -e .
```

## Quick Start

### Simple Usage (2 lines)

```python
from valtec_tts import TTS

tts = TTS()  # Auto-downloads model from Hugging Face if not cached
tts.speak("Xin chào các bạn", output_path="hello.wav")
```

### Get Audio Array

```python
from valtec_tts import TTS

tts = TTS()
audio, sr = tts.synthesize("Xin chào các bạn")
```

### Choose Speaker

```python
from valtec_tts import TTS

tts = TTS()
print(tts.list_speakers())  # ['NF', 'SF', 'NM1', 'SM', 'NM2']

# NF = Northern Female, SF = Southern Female
tts.speak("Xin chào", speaker="NF", output_path="hello.wav")

# NM1/NM2 = Northern Male, SM = Southern Male  
tts.speak("Xin chào", speaker="NM1", output_path="hello.wav")
```

### Adjust Speed

```python
tts.speak("Nói nhanh hơn", speed=0.8, output_path="fast.wav")   # Faster
tts.speak("Nói chậm hơn", speed=1.3, output_path="slow.wav")    # Slower
```

### Select Device (CUDA/CPU)

```python
tts = TTS(device="cuda")  # Use GPU
tts = TTS(device="cpu")   # Use CPU
tts = TTS()               # Auto-detect (default)
```

### Use Local Model

```python
tts = TTS(model_path="./pretrained")
tts.speak("Xin chào", output_path="hello.wav")
```

## Command Line

```bash
# Single text synthesis
python infer.py --text "Xin chào các bạn" --speaker NF --output hello.wav

# Try different voices
python infer.py --text "Xin chào các bạn" --speaker NM1 --output hello_nm1.wav
python infer.py --text "Xin chào các bạn" --speaker SM --output hello_sm.wav

# Interactive mode
python infer.py --interactive

# Batch processing from file
python infer.py --input_file texts.txt --output_dir ./outputs
```

## Gradio Demo

```bash
python demo_gradio.py
```

Then open your browser at `http://localhost:7860`

## Available Speakers

The pretrained model includes **5 Vietnamese voices** with regional accents:

| Speaker | Region | Gender | Code | Description |
|---------|--------|--------|------|-------------|
| **NF** | 🌆 Northern (Miền Bắc) | 👩 Female | `NF` | Northern Female voice |
| **SF** | 🌾 Southern (Miền Nam) | 👩 Female | `SF` | Southern Female voice |
| **NM1** | 🌆 Northern (Miền Bắc) | 👨 Male | `NM1` | Northern Male voice 1 |
| **SM** | 🌾 Southern (Miền Nam) | 👨 Male | `SM` | Southern Male voice |
| **NM2** | 🌆 Northern (Miền Bắc) | 👨 Male | `NM2` | Northern Male voice 2 |

**Usage Example:**
```python
tts.speak("Xin chào", speaker="NF")   # Northern Female
tts.speak("Xin chào", speaker="NM1")  # Northern Male 1
tts.speak("Xin chào", speaker="SM")   # Southern Male
```

## Synthesis Parameters

- `speed` (default: 1.0): Speech speed
  - '< 1.0 = faster'
  - '> 1.0 = slower'
- `noise_scale` (default: 0.667): Controls variability in generated speech
- `noise_scale_w` (default: 0.8): Controls duration variability
- `sdp_ratio` (default: 0.0): Stochastic Duration Predictor ratio
  - 0.0 = deterministic
  - 1.0 = fully stochastic

## Model Auto-Download

When you first use `TTS()` without specifying a model path, the pretrained model will be automatically downloaded from Hugging Face and cached locally:

- Windows: `%LOCALAPPDATA%\valtec_tts\models\`
- Linux/Mac: `~/.cache/valtec_tts/models/`

To use a custom Hugging Face repository:

```python
tts = TTS(hf_repo="valtecAI-team/valtec-tts-pretrained")
```

## Project Structure

```
valtec-tts/
├── valtec_tts/           # Main package
│   ├── __init__.py
│   └── tts.py            # Simple TTS API
├── src/
│   ├── models/           # Neural network models
│   ├── text/             # Text processing
│   ├── vietnamese/       # Vietnamese-specific modules
│   ├── nn/               # Neural network utilities
│   └── utils/            # General utilities
├── pretrained/           # Local pretrained models
│   └── onnx/             # ONNX export models
├── deployments/          # Production deployments
│   ├── edge/             # Edge/lightweight deployment (ONNX)
│   ├── web/              # Browser-based demo
│   └── android/          # Android mobile app
├── infer.py              # Inference script
├── app.py                # Gradio web demo
├── export_full_onnx.py   # ONNX export script
└── README.md
```

## 📱 Deployment Options

### Edge/Lightweight Deployment

**ONNX Runtime** - Optimized for edge devices and lightweight deployment

```bash
cd deployments/edge
python inference.py --text "Hello Vietnam" --speaker 2
```

Features:
- Auto-downloads models from HuggingFace Hub
- ~165MB total model size
- CPU & GPU support
- See [deployments/edge/README.md](deployments/edge/README.md)

### Web Demo

**Browser-based** - No backend server required

```bash
# Serve from project root
npx -y http-server . -p 8080
# Open: http://localhost:8080/deployments/web/
```

Features:
- Full ONNX Runtime Web
- Runs entirely in browser
- See [deployments/web/README.md](deployments/web/README.md)

### Android App

**Mobile deployment** - On-device TTS

```bash
cd deployments/android
./gradlew assembleDebug
```

Features:
- ONNX Runtime Mobile
- ~185MB APK (models included)
- Offline-capable
- See [deployments/android/README.md](deployments/android/README.md)

### API Integration

```python
# Using HuggingFace Spaces API
import requests

response = requests.post(
    "https://valtecai-team-valtec-vietnamese-tts.hf.space/api/synthesize",
    json={"text": "Xin chào", "speaker": "female"}
)
```

## License

This project is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Creative Commons Attribution-NonCommercial 4.0 International).

- You may use, share, and adapt this project for non-commercial purposes only.
- Commercial use is strictly prohibited without prior written permission.
- You must give appropriate credit to the original authors.

## Acknowledgments

- Vietnamese phonemization support
- Valtec team for model training and development
