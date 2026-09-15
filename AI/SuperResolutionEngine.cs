using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using Enhancer.Utils;

namespace Enhancer.AI
{
    public class SuperResolutionEngine : OnnxModelBase
    {
        private bool _isMockMode = false;
        private int _scaleFactor = 4; // Assuming x4 upscaling for Real-ESRGAN

        public SuperResolutionEngine(string modelPath) : base(modelPath)
        {
            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"[!] SuperResolution model not found at {modelPath}. Running in MOCK Mode (OpenCV Resize).");
                _isMockMode = true;
            }
        }

        public Mat Upscale(Mat inputFrame)
        {
            if (_isMockMode)
            {
                // Mock Mode: Just resize the image
                Mat outputFrame = new Mat();
                Cv2.Resize(inputFrame, outputFrame, new Size(inputFrame.Width * _scaleFactor, inputFrame.Height * _scaleFactor), 0, 0, InterpolationFlags.Cubic);
                return outputFrame;
            }

            // Real ONNX Inference
            int width = inputFrame.Width;
            int height = inputFrame.Height;
            
            // 1. Convert Mat to Float Array
            float[] tensorData = TensorUtils.ConvertMatToTensorArray(inputFrame);
            
            // 2. Create Input Tensor (BatchSize: 1, Channels: 3, Height, Width)
            var inputTensor = new DenseTensor<float>(tensorData, new[] { 1, 3, height, width });
            
            // 3. Prepare Inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 4. Run Inference
            using var results = _session.Run(inputs);
            var outputResult = results.First();
            var outputTensor = outputResult.AsTensor<float>();

            // 5. Convert Output Tensor back to Mat
            int outHeight = height * _scaleFactor;
            int outWidth = width * _scaleFactor;
            
            float[] outputData = outputTensor.ToArray();
            Mat highResMat = TensorUtils.ConvertTensorArrayToMat(outputData, outWidth, outHeight);

            return highResMat;
        }
    }
}
