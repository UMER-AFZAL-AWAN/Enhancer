using System;
using OpenCvSharp;

namespace Enhancer.AI
{
    public class AIPipelineController : IDisposable
    {
        private SuperResolutionEngine _srEngine;
        private EffectInjector _fxEngine;
        private OpticalFlowEngine _flowEngine;
        
        private Mat _previousAura;
        private int _powerModeFrameCounter = 0;

        public AIPipelineController()
        {
            _srEngine = new SuperResolutionEngine("Models/realesrgan-x4.onnx");
            _fxEngine = new EffectInjector("Models/yolov8n-seg.onnx");
            _flowEngine = new OpticalFlowEngine("Models/raft.onnx");
            Console.WriteLine("AIPipelineController initialized.");
        }

        public Mat ProcessFrame(Mat inputFrame, double timestampSeconds)
        {
            Mat processedFrame = inputFrame.Clone();
            bool isPowerMode = timestampSeconds >= 12.0 && timestampSeconds <= 15.0;

            if (isPowerMode)
            {
                // 1. Calculate Optical Flow
                Mat flow = _flowEngine.ComputeFlow(inputFrame);

                // 2. Extract player mask
                Mat playerMask = _fxEngine.GetPlayerMask(inputFrame);
                
                Mat fxFrame;

                // 3. Temporal Consistency Strategy: 
                // Run heavy generative AI every 5 frames. Otherwise, warp the previous aura.
                if (_powerModeFrameCounter % 5 == 0 || flow == null || _previousAura == null)
                {
                    fxFrame = _fxEngine.ApplyAnimeAura(inputFrame, playerMask);
                }
                else
                {
                    // Warp the old aura using optical flow to perfectly match movement!
                    fxFrame = _flowEngine.WarpEffect(_previousAura, flow);
                }

                // Cache aura for next frame
                if (_previousAura != null) _previousAura.Dispose();
                _previousAura = fxFrame.Clone();

                // 4. Blend aura over original frame
                Mat blendedFrame = _fxEngine.Blend(processedFrame, fxFrame, playerMask);

                // Add text overlay
                Cv2.PutText(blendedFrame, "POWER MODE (WITH OPTICAL FLOW)", new Point(50, 100), HersheyFonts.HersheyComplex, 1.2, Scalar.Red, 2);
                
                // Cleanup intermediate mats
                processedFrame.Dispose();
                fxFrame.Dispose();
                playerMask.Dispose();
                if (flow != null) flow.Dispose();

                processedFrame = blendedFrame;
                _powerModeFrameCounter++;
            }
            else
            {
                _powerModeFrameCounter = 0;
                if (_previousAura != null)
                {
                    _previousAura.Dispose();
                    _previousAura = null;
                }
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
            _fxEngine?.Dispose();
            _flowEngine?.Dispose();
            _previousAura?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
