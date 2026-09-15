using System;
using OpenCvSharp;

namespace Enhancer.Utils
{
    public static class TensorUtils
    {
        public static float[] ConvertMatToTensorArray(Mat image)
        {
            // Converts HWC (BGR) OpenCV Mat (Byte) to NCHW (RGB) float array normalized to [0, 1]
            int height = image.Height;
            int width = image.Width;
            int channels = image.Channels();
            
            float[] tensorArray = new float[channels * height * width];

            unsafe
            {
                byte* data = (byte*)image.DataPointer;
                int stride = (int)image.Step();

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        // OpenCV is BGR
                        byte b = data[y * stride + x * channels + 0];
                        byte g = data[y * stride + x * channels + 1];
                        byte r = data[y * stride + x * channels + 2];

                        // Target is NCHW (RGB)
                        tensorArray[0 * height * width + y * width + x] = r / 255.0f;
                        tensorArray[1 * height * width + y * width + x] = g / 255.0f;
                        tensorArray[2 * height * width + y * width + x] = b / 255.0f;
                    }
                }
            }

            return tensorArray;
        }

        public static Mat ConvertTensorArrayToMat(float[] tensorArray, int width, int height)
        {
            // Converts NCHW (RGB) float array back to HWC (BGR) OpenCV Mat (Byte)
            int channels = 3;
            Mat image = new Mat(height, width, MatType.CV_8UC3);

            unsafe
            {
                byte* data = (byte*)image.DataPointer;
                int stride = (int)image.Step();

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float r = tensorArray[0 * height * width + y * width + x];
                        float g = tensorArray[1 * height * width + y * width + x];
                        float b = tensorArray[2 * height * width + y * width + x];

                        // Clamp to [0, 1] and map to [0, 255]
                        byte rByte = (byte)Math.Clamp(Math.Round(r * 255.0), 0, 255);
                        byte gByte = (byte)Math.Clamp(Math.Round(g * 255.0), 0, 255);
                        byte bByte = (byte)Math.Clamp(Math.Round(b * 255.0), 0, 255);

                        // OpenCV is BGR
                        data[y * stride + x * channels + 0] = bByte;
                        data[y * stride + x * channels + 1] = gByte;
                        data[y * stride + x * channels + 2] = rByte;
                    }
                }
            }

            return image;
        }
    }
}
