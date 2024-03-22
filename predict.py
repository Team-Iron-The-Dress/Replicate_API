# Prediction interface for Cog ⚙️
# https://cog.run/python
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
from cog import BasePredictor, Input, Path, File


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = pipe
        #initalizing the model
        pipe = AutoPipelineForInpainting.from_pretrained('stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant="fp16")
        pipe.to('cuda')
        pipe.load_lora_weights("./sdxl_lora _latest.safetensors")
        
        
        
    def predict(
        self,
        image: Path = Input(description="Input to remove wrinkles"),
        mask_image: Path = Input(description="Mask image of your ground truth image"),
    ) -> File:
        
        """Run a model"""
        #opening images
        input_image = Image.open(image)
        mask = Image.open(mask_image)
        
        # running the model 
        output_image = self.pipe(
            prompt="remove wrinkles from the cloth, maintain cloth color consistency, don't operate on cloth folds, avoid changing cloth prints",
            negative_prompt="wrinkled, low quality texture, degenerated texture, unintelligible text, overexposed, worst quality, low quality, jpeg artifacts, ugly, deformed, blurry",
            image=input_image,
            mask_image=mask,
            num_inference_steps=5,
            strength=0.5,
            guidance_scale=1
        )
        
        #returning image
        return File(output_image)
        
