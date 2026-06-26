# Offline Generative Video Enhancer

A fully offline, local C# .NET 8 pipeline for enhancing football video footage using ONNX AI models.

## Features (Phase 1)
- **Modular Pipeline:** Setup using `Microsoft.ML.OnnxRuntime.Gpu` for fast inference.
- **Video Processing:** OpenCV-powered `VideoProcessor` for reliable frame decoding/encoding.
- **AI Routing Engine:** Timestamp-aware `AIPipelineController` for deciding when to run standard Super-Resolution vs. "Anime Power Mode" generation.
- **CUDA Support:** Native GPU hardware acceleration hooks implemented via `OnnxModelBase`.

## Current Status
- ✅ Project Foundation & Dependency Setup
- ✅ Video I/O Pipeline implementation
- ⏳ Phase 2: Super-Resolution Integration (Pending)
- ⏳ Phase 3: YOLOv8 Segmentation & Styling (Pending)
- ⏳ Phase 4: Optical Flow Consistency (Pending)

## How to run
1. Ensure you have the .NET 8 SDK installed.
2. Ensure you have an NVIDIA GPU with CUDA drivers installed (for hardware acceleration).
3. Build and run the project:
   ```bash
   dotnet build
   dotnet run -- input.mp4 output.mp4
   ```
*(If `input.mp4` does not exist, the app will automatically generate a mock video for testing).*
