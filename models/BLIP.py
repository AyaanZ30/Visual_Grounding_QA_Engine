import os 
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

class BLIPVisualReasoning:
    def __init__(self, device = "cpu", use_fp16 = False, model_id = "Salesforce/blip-vqa-base"):
        self.device = device
        
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.BLIP = BlipForQuestionAnswering.from_pretrained(model_id)
        
        if use_fp16 and device == 'cuda':
            self.BLIP = self.BLIP.half()
        
        self.BLIP.to(self.device)
        self.BLIP.eval()
    
    @torch.no_grad()
    def answer(self, image : Image, question : str) -> str:
        inputs = self.processor(image, question, return_tensors = 'pt')
        inputs = {k : v.to(self.device) for k, v in inputs.items()}
        
        output_ids = self.BLIP.generate(**inputs, max_length = 20)
        
        response = self.processor.decode(output_ids[0], skip_special_tokens = True)
        return response
        