import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from model import AttributeRecognitionModel  # Impor model dari model.py

# Definisi label untuk 6 kelas
label_col = np.array(['Glasses', 'NoGlasses', 'Helmet', 'NoHelmet', 'Mask', 'NoMask', 'Rompi', 'NoRompi'])

def preprocess_image(image_path, resize=(224, 224)):
    # Buka gambar dengan PIL (sesuai dengan transformasi pelatihan)
    image = Image.open(image_path).convert('RGB')
    
    # Terapkan transformasi yang sama seperti saat pelatihan
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Proses gambar
    img_tensor = transform(image)
    
    return img_tensor

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def perform_inference(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Proses gambar
    normalized_image = preprocess_image(image_path)
    normalized_image_tensor = normalized_image.to(device)
    normalized_image_tensor = normalized_image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(normalized_image_tensor)

    # Hitung probabilitas
    predicted_probs = output.cpu().numpy().astype(float)
    predicted_probs = sigmoid(predicted_probs)
    
    # Tentukan label yang diprediksi (probabilitas > 0.5)
    predicted_results = predicted_probs[0] > 0.5
    pos = np.where(predicted_results == 1)[0]
    
    return {"labels": label_col[pos], "prob": predicted_probs[0][pos]}

def perform_inference_with_visualization(model, image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Buat gambar putih sebagai latar
    white_image = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # Muat gambar orang
    person_image = cv2.imread(image_path)
    person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)

    # Ubah ukuran gambar orang agar muat di latar putih
    person_image = cv2.resize(person_image, (128, 64))

    # Hitung posisi untuk menempatkan gambar di tengah
    y_offset = (256 - person_image.shape[0]) // 2
    x_offset = (256 - person_image.shape[1]) // 2

    # Tempatkan gambar orang di latar putih
    white_image[y_offset:y_offset + person_image.shape[0], x_offset:x_offset + person_image.shape[1]] = person_image

    # Proses gambar untuk inferensi
    normalized_image = preprocess_image(image_path)
    normalized_image_tensor = normalized_image.to(device)
    normalized_image_tensor = normalized_image_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(normalized_image_tensor)

    # Hitung probabilitas
    predicted_probs = output.cpu().numpy().astype(float)
    predicted_probs = sigmoid(predicted_probs)

    # Tentukan label yang diprediksi
    predicted_results = predicted_probs[0] > 0.5
    pos = np.where(predicted_results == 1)[0]
    labels = label_col[pos]
    probs = predicted_probs[0][pos]

    # Tambahkan teks label ke gambar
    y_text = y_offset + person_image.shape[0] + 20  # Posisi teks di bawah gambar
    for i, (label, prob) in enumerate(zip(labels, probs)):
        text = f"{label}: {prob:.2f}"
        cv2.putText(white_image, text, (x_offset, y_text + i * 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Tampilkan gambar hasil
    plt.imshow(white_image)
    plt.axis('off')
    plt.show()

    # Simpan gambar hasil
    cv2.imwrite(output_path, cv2.cvtColor(white_image, cv2.COLOR_RGB2BGR))

    return {"labels": labels, "prob": probs}

def main():
    parser = argparse.ArgumentParser(description='Perform inference on an image using a trained PyTorch model.')
    parser.add_argument('--model_path', type=str, default='models/safety_gear_model.pth', 
                        help='Path to the trained PyTorch model file')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Path to the input image for inference')
    parser.add_argument('--output_path', type=str, default='output/result.jpg', 
                        help='Path to save the output image with labels')
    args = parser.parse_args()
    
    print(f"Model path: {args.model_path}")
    print(f"Image path: {args.image_path}")
    print(f"Output path: {args.output_path}")

    # Muat model
    model = AttributeRecognitionModel()
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    
    # Lakukan inferensi dengan visualisasi
    results = perform_inference_with_visualization(model, args.image_path, args.output_path)
    
    print("Predicted results:", results)

if __name__ == "__main__":
    main()