from llama_cpp import Llama

class LlamaCPPInvocationLayer:
    def __init__(self, model_path, use_gpu=False, max_length=512):
        self.model = Llama(model_path=model_path, n_gpu_layers=1 if use_gpu else 0)
        self.max_length = max_length

    def __call__(self, prompt: str):
        response = self.model(prompt=prompt, max_tokens=self.max_length)
        return response.get('choices', [{}])[0].get('text', '')
