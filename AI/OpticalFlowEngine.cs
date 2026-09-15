using System;
using System.IO;
using OpenCvSharp;

namespace Enhancer.AI
{
    public class OpticalFlowEngine : OnnxModelBase
    {
        private bool _isMockMode = false;
        private Mat _prevFrameGray;

        public OpticalFlowEngine(string raftModelPath) : base(raftModelPath)
        {
            if (!File.Exists(raftModelPath))
            {
                Console.WriteLine($"[!] RAFT model not found at {raftModelPath}. Running in MOCK Mode (OpenCV Farneback Flow).");
                _isMockMode = true;
            }
        }

        public Mat ComputeFlow(Mat currentFrame)
        {
            Mat currentGray = new Mat();
            Cv2.CvtColor(currentFrame, currentGray, ColorConversionCodes.BGR2GRAY);

            Mat flow = new Mat();

            if (_prevFrameGray == null || _prevFrameGray.Empty())
            {
                // First frame, no flow
                _prevFrameGray = currentGray.Clone();
                return null;
            }

            if (_isMockMode)
            {
                // OpenCV Farneback Optical Flow (Mock for RAFT)
                Cv2.CalcOpticalFlowFarneback(
                    _prevFrameGray, currentGray, flow, 
                    pyrScale: 0.5, levels: 3, winsize: 15, 
                    iterations: 3, polyN: 5, polySigma: 1.2, 
                    flags: 0);
            }
            else
            {
                // Real RAFT ONNX Inference logic goes here
            }

            // Save current frame for next iteration
            _prevFrameGray.Dispose();
            _prevFrameGray = currentGray.Clone();

            return flow;
        }

        public Mat WarpEffect(Mat previousEffect, Mat flow)
        {
            if (flow == null || flow.Empty()) return previousEffect;

            Mat warpedEffect = new Mat();
            
            // Generate mapping matrices from the flow vectors
            Mat mapX = new Mat(flow.Size(), MatType.CV_32FC1);
            Mat mapY = new Mat(flow.Size(), MatType.CV_32FC1);

            unsafe
            {
                float* flowData = (float*)flow.DataPointer;
                float* xData = (float*)mapX.DataPointer;
                float* yData = (float*)mapY.DataPointer;

                int cols = flow.Cols;
                int rows = flow.Rows;

                for (int y = 0; y < rows; y++)
                {
                    for (int x = 0; x < cols; x++)
                    {
                        int idxFlow = (y * cols + x) * 2;
                        int idxMap = y * cols + x;

                        // Calculate where this pixel moved TO
                        xData[idxMap] = x + flowData[idxFlow];
                        yData[idxMap] = y + flowData[idxFlow + 1];
                    }
                }
            }

            // Remap the previous effect using the flow map
            Cv2.Remap(previousEffect, warpedEffect, mapX, mapY, InterpolationFlags.Linear);

            mapX.Dispose();
            mapY.Dispose();

            return warpedEffect;
        }
    }
}
