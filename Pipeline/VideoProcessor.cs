using System;
using OpenCvSharp;
using Enhancer.AI;

namespace Enhancer.Pipeline
{
    public class VideoProcessor
    {
        private readonly string _inputPath;
        private readonly string _outputPath;

        public VideoProcessor(string inputPath, string outputPath)
        {
            _inputPath = inputPath;
            _outputPath = outputPath;
        }

        public void Process()
        {
            using var capture = new VideoCapture(_inputPath);
            if (!capture.IsOpened())
            {
                Console.WriteLine($"Error: Cannot open video file {_inputPath}");
                return;
            }

            int fps = (int)Math.Round(capture.Fps);
            int width = capture.FrameWidth;
            int height = capture.FrameHeight;
            int totalFrames = capture.FrameCount;

            Console.WriteLine($"Input Video: {width}x{height} @ {fps}fps. Total Frames: {totalFrames}");

            // Note: In a real scenario with SR, the output width/height would be scaled up (e.g., width*4).
            // For Phase 1, we keep it the same.
            using var writer = new VideoWriter(_outputPath, FourCC.MP4V, fps, new Size(width, height));
            if (!writer.IsOpened())
            {
                Console.WriteLine($"Error: Cannot open output video file {_outputPath}");
                return;
            }

            using var aiController = new AIPipelineController();
            using var frame = new Mat();

            int frameCount = 0;
            while (capture.Read(frame) && !frame.Empty())
            {
                double timestampSeconds = capture.PosMsec / 1000.0;
                
                using Mat processedFrame = aiController.ProcessFrame(frame, timestampSeconds);
                writer.Write(processedFrame);

                frameCount++;
                if (frameCount % 30 == 0)
                {
                    Console.WriteLine($"Processed {frameCount}/{totalFrames} frames...");
                }
            }

            Console.WriteLine($"Done. Processed {frameCount} frames. Output saved to {_outputPath}");
        }
    }
}
