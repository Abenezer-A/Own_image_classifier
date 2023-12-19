import argparse
import json
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

def input_args():
    parser = argparse.ArgumentParser(description="Prediction using model.")
    parser.add_argument("img_dir", type=str, help="path to image.")
    parser.add_argument("chk_dir", type=str, help="Path to checkpoint.")
    parser.add_argument("--cat_to_name", type=str, default="none", help="JSON file name to be provided.")
    parser.add_argument("--top_k", type=int, default=5, help="Top probability to be displayed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU.")
    return parser.parse_args()

def load_checkpoint(file_path):
    if args.gpu:
    # Checkpoint for when using GPU
        checkpoint = torch.load(filepath)
    else:
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    if checkpoint['model'] == "VGG16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)

    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path: str) -> np.ndarray:
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Processed image as a Numpy array.
    """

    # Load PIL image
    pil_image = Image.open(f"{image_path}")

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transforms to the image
    transformed_image = transform(pil_image)

    # Convert PIL image to Numpy array
    array_image = np.array(transformed_image)

    return array_image
    
def predict(image_path, model, topk=5):
    
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    # Set device to cuda or gpu
    device = torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    # Move model to the chosen device
    model.to(device)

    # Process image and convert to torch tensor
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Set model to evaluation mode and disable gradients
    model.eval()
    with torch.no_grad():
        output = model.forward(image)

    # Get probabilities and class indices
    output_prob = torch.exp(output)
    probs, indeces = output_prob.topk(topk)

    # Move results back to CPU for easier access
    probs = probs.to('cpu').numpy().tolist()[0]
    indeces = indeces.to('cpu').numpy().tolist()[0]

    # Map indices to class names
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[item] for item in indeces]

    return probs, classes

def cat_to_name(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    

def main():
    
    args = input_args()
    model = load_checkpoint(args.chk_dir)
    cat_to_name = cat_to_name(args.cat_to_name)
    probs, classes = predict(args.img_dir, model, args.top_k)

    class_name = [cat_to_name[str(i)] for i in classes]

    print(f"Class_name: {class_name}")
    print(f"Probability (%): {probs*100:.3f}")


          
if __name__ == "__main__":
  main()


# command line usage: 
# python predict.py image_path/image.jpg checkpoint.pth --gpu
# python predict.py image_path/image.jpg checkpoint.pth --cat_to_name cat_to_name.json --top_k 5 --gpu
