using System;
using OpenCvSharp;

namespace Enhancer.AI
{
    public class AIPipelineController : IDisposable
    {
        private SuperResolutionEngine _srEngine;
        // private EffectInjector _fxEngine;
        
        public AIPipelineController()
        {
            _srEngine = new SuperResolutionEngine("Models/realesrgan-x4.onnx");
            // _fxEngine = new EffectInjector("Models/yolov8n-seg.onnx", "Models/style-transfer.onnx");
            Console.WriteLine("AIPipelineController initialized. (Effect Models pending in Phase 3)");
        }

        public Mat ProcessFrame(Mat inputFrame, double timestampSeconds)
        {
            Mat processedFrame = inputFrame.Clone();
            bool isPowerMode = timestampSeconds >= 12.0 && timestampSeconds <= 15.0;

            if (isPowerMode)
            {
                Cv2.Rectangle(processedFrame, new Point(50, 50), new Point(inputFrame.Width - 50, inputFrame.Height - 50), Scalar.Red, 5);
                Cv2.PutText(processedFrame, "POWER MODE ACTIVE", new Point(100, 100), HersheyFonts.HersheyComplex, 1.5, Scalar.Red, 2);
            }
            else
            {
                Cv2.PutText(processedFrame, "NORMAL SR PROCESSING", new Point(100, 100), HersheyFonts.HersheyComplex, 1.5, Scalar.Green, 2);
            }

            // Phase 2: Upscale frame via Real-ESRGAN
            Mat upscaledFrame = _srEngine.Upscale(processedFrame);

            // Free the intermediate frame
            processedFrame.Dispose();

            return upscaledFrame;
        }

        public void Dispose()
        {
            _srEngine?.Dispose();
            // _fxEngine?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
