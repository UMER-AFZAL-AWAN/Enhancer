using System;
using System.IO;
using Enhancer.Pipeline;
using OpenCvSharp;

namespace Enhancer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=============================================");
            Console.WriteLine("    Offline Generative Video Enhancer        ");
            Console.WriteLine("=============================================\n");

            string inputPath = args.Length > 0 ? args[0] : "input.mp4";
            string outputPath = args.Length > 1 ? args[1] : "output.mp4";

            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"[!] Input file not found: {inputPath}");
                Console.WriteLine("Generating a dummy input video for testing...");
                GenerateDummyVideo(inputPath);
            }

            Console.WriteLine($"[+] Input: {inputPath}");
            Console.WriteLine($"[+] Output: {outputPath}");
            Console.WriteLine("[+] Starting processing pipeline...\n");

            var processor = new VideoProcessor(inputPath, outputPath);
            processor.Process();

            Console.WriteLine("\n[+] Processing complete!");
        }

        static void GenerateDummyVideo(string path)
        {
            int width = 640;
            int height = 480;
            int fps = 30;
            int totalFrames = 30 * 20; // 20 seconds

            using var writer = new VideoWriter(path, FourCC.MP4V, fps, new Size(width, height));
            using var frame = new Mat(height, width, MatType.CV_8UC3, new Scalar(0, 0, 0));

            for (int i = 0; i < totalFrames; i++)
            {
                double seconds = i / (double)fps;
                frame.SetTo(new Scalar(50, 50, 50)); // Dark grey background
                
                // Draw a moving circle
                int x = (int)(width / 2 + Math.Sin(seconds * 2) * 100);
                int y = (int)(height / 2 + Math.Cos(seconds * 2) * 100);
                
                Cv2.Circle(frame, new Point(x, y), 30, Scalar.Yellow, -1);
                Cv2.PutText(frame, $"Time: {seconds:F2}s", new Point(20, 40), HersheyFonts.HersheySimplex, 1, Scalar.White, 2);

                writer.Write(frame);
            }
            Console.WriteLine($"[+] Generated dummy video at {path}");
        }
    }
}
