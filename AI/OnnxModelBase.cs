using System;
using Microsoft.ML.OnnxRuntime;

namespace Enhancer.AI
{
    public abstract class OnnxModelBase : IDisposable
    {
        protected InferenceSession _session;
        protected string _modelPath;

        public OnnxModelBase(string modelPath, int gpuDeviceId = 0)
        {
            _modelPath = modelPath;
            InitializeSession(gpuDeviceId);
        }

        private void InitializeSession(int gpuDeviceId)
        {
            var options = new SessionOptions();
            
            // Enable CUDA
            try
            {
                options.AppendExecutionProvider_CUDA(gpuDeviceId);
                Console.WriteLine($"Initialized {_modelPath} with CUDA Execution Provider.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not initialize CUDA provider for {_modelPath}. Falling back to CPU. Error: {ex.Message}");
            }

            _session = new InferenceSession(_modelPath, options);
        }

        public void Dispose()
        {
            _session?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
