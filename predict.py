from cog import BasePredictor, Path, Input
import torch

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions 
efficient"""
        self.model = torch.load("CodeLlama-7b/consolidated.00.pth")

    def predict(self,
            image: Path = Input(description="Image to enlarge"),
            scale: float = Input(description="Factor to scale image by", 
default=1.5)
    ) -> Path:
        """Run a single prediction on the model"""
        # ... pre-processing ...
        output = self.model(image)
        # ... post-processing ...
        return output
