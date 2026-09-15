using System;
using System.IO;
using OpenCvSharp;
using Enhancer.Utils;

namespace Enhancer.AI
{
    public class EffectInjector : OnnxModelBase
    {
        private bool _isMockMode = false;
        private int _frameCount = 0;

        public EffectInjector(string segmentationModelPath) : base(segmentationModelPath)
        {
            if (!File.Exists(segmentationModelPath))
            {
                Console.WriteLine($"[!] YOLOv8 model not found at {segmentationModelPath}. Running in MOCK Mode (OpenCV Contour detection).");
                _isMockMode = true;
            }
        }

        public Mat GetPlayerMask(Mat inputFrame)
        {
            Mat mask = new Mat(inputFrame.Size(), MatType.CV_8UC1, new Scalar(0));

            if (_isMockMode)
            {
                // Mock Mode: Detect the bright yellow circle we generated in the mock video
                Mat hsv = new Mat();
                Cv2.CvtColor(inputFrame, hsv, ColorConversionCodes.BGR2HSV);
                
                // Yellow color range
                Scalar lowerBound = new Scalar(20, 100, 100);
                Scalar upperBound = new Scalar(30, 255, 255);
                
                Cv2.InRange(hsv, lowerBound, upperBound, mask);
                
                // Cleanup mock mask
                Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(5, 5));
                Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
                return mask;
            }

            // Real YOLOv8 Inference logic goes here...
            // Extract tensor, run inference, decode mask tensors to Mat.
            // For brevity in Phase 3 skeleton, assuming mask extraction logic.
            
            return mask;
        }

        public Mat ApplyAnimeAura(Mat inputFrame, Mat mask)
        {
            _frameCount++;

            // Create an aura mask by dilating the original player mask
            Mat auraMask = new Mat();
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(25, 25));
            Cv2.Dilate(mask, auraMask, kernel);

            // Subtract inner mask so aura is only on the edges
            Mat edgesOnly = new Mat();
            Cv2.BitwiseXor(auraMask, mask, edgesOnly);
            
            // Blur it to make it soft and glowing
            Cv2.GaussianBlur(edgesOnly, edgesOnly, new Size(21, 21), 0);

            // Generate an Electric Blue / Fire aura color
            Mat coloredAura = new Mat(inputFrame.Size(), MatType.CV_8UC3, new Scalar(255, 150, 0)); // Cyan/Blue aura

            // Apply a slight oscillation to simulate pulsing energy
            double pulse = (Math.Sin(_frameCount * 0.5) + 1.0) / 2.0; // 0.0 to 1.0
            coloredAura = coloredAura * (0.5 + 0.5 * pulse);

            // Mask the color
            Mat fxFrame = new Mat();
            coloredAura.CopyTo(fxFrame, edgesOnly);

            return fxFrame;
        }

        public Mat Blend(Mat original, Mat fxFrame, Mat mask)
        {
            Mat result = new Mat();
            // Additive blending for glowing aura effect
            Cv2.AddWeighted(original, 1.0, fxFrame, 0.8, 0, result);
            return result;
        }
    }
}
