import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

class DenseNetTeacher:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = models.densenet121(pretrained=False)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 14)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device).eval()

        self.preproc = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
        ])

    def infer(self, image_path):
        img = Image.open(image_path).convert('RGB')
        x = self.preproc(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().squeeze()  # shape (14,)
