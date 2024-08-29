import os
import pydicom
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from skimage import filters
from sklearn.model_selection import train_test_split
from PIL import Image


def preprocess_and_save_images(csv_file, root_dir, output_dir, size=224, test_size=0.2):
    print("Lendo o arquivo CSV...")
    data = pd.read_csv(csv_file, sep=';')
    print(f"Total de entradas no CSV: {len(data)}")

    data = data[data['Bi-Rads'].notnull()]
    print(f"Total de entradas após remover Bi-Rads nulos: {len(data)}")

    data['Label'] = data['Bi-Rads'].apply(lambda x: 0 if x in ['1', '2'] else 1)
    print("Labels atribuídos com base no Bi-Rads.")

    # Convertendo a coluna 'File Name' para string
    data['File Name'] = data['File Name'].astype(str)

    # Dividir os dados em conjuntos de treino e teste
    print("Dividindo os dados em conjuntos de treino e teste...")
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['Label'], random_state=42)
    print(f"Total de imagens no treino: {len(train_data)}, no teste: {len(test_data)}")

    # Criar um dicionário para mapear o nome dos arquivos aos rótulos
    train_labels = dict(zip(train_data['File Name'], train_data['Label']))
    test_labels = dict(zip(test_data['File Name'], test_data['Label']))

    # Função para processar e salvar imagens
    def process_and_save(dataset, split, labels_dict):
        print(f"Processando o conjunto de {split}...")

        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for label in [0, 1]:
            label_output_dir = os.path.join(split_output_dir, str(label))
            os.makedirs(label_output_dir, exist_ok=True)

        for file_name in dataset['File Name']:
            dicom_file = file_name + '.dcm'
            dicom_path = os.path.join(root_dir, dicom_file)

            if not os.path.isfile(dicom_path):
                print(f"Arquivo DICOM {dicom_file} não encontrado no diretório, pulando...")
                continue

            print(f"Processando imagem DICOM: {dicom_file}")
            image = pydicom.dcmread(dicom_path).pixel_array

            # Pré-processamento da imagem
            image = histogram_equalization(image)
            bin_img = preprocess_before_crop(image)
            img_mask = get_mask_of_largest_connected_component(bin_img)
            cropped_img = crop_img(image, img_mask)

            # Converter a imagem para PIL antes de aplicar as transformações
            image = Image.fromarray(cropped_img)

            # Redimensionar a imagem
            image = image.resize((size, size))

            # Converter a imagem de volta para numpy array
            image = np.array(image)

            # Obter o rótulo a partir do dicionário
            label = labels_dict[file_name]
            print(f"Rótulo para {file_name}: {label}")

            # Salvar a imagem no diretório correspondente
            label_dir = os.path.join(split_output_dir, str(label))
            output_path = os.path.join(label_dir, f"{file_name}.png")
            cv2.imwrite(output_path, image)
            print(f"Imagem salva em: {output_path}")

    # Processar e salvar imagens para treino e teste
    process_and_save(train_data, 'train', train_labels)
    process_and_save(test_data, 'test', test_labels)


def histogram_equalization(img):
    if np.max(img) > 0:
        m = int(np.max(img))
        hist = np.histogram(img, bins=m + 1, range=(0, m + 1))[0]
        hist = hist / img.size
        cdf = np.cumsum(hist)
        s_k = (255 * cdf).astype(np.uint8)
        img_new = np.array([s_k[i] for i in img.ravel()]).reshape(img.shape)
        return img_new
    else:
        return img


def preprocess_before_crop(img):
    threshold = filters.threshold_sauvola(img)
    binary_img = (img > threshold) * 1
    kernel = np.ones((5, 5), np.uint8)
    binary_img = binary_img.astype('uint8')
    binary_img = cv2.erode(binary_img, kernel, iterations=2)
    return binary_img


def get_mask_of_largest_connected_component(img_mask):
    mask, num_labels = scipy.ndimage.label(img_mask)
    mask_pixels_dict = {i: np.sum(mask == i) for i in range(1, num_labels + 1)}
    largest_mask_index = max(mask_pixels_dict, key=mask_pixels_dict.get)
    largest_mask = (mask == largest_mask_index)
    return largest_mask


def crop_img(img, img_mask):
    farest_pixel = np.max(list(zip(*np.where(img_mask == 1))), axis=0)
    nearest_pixel = np.min(list(zip(*np.where(img_mask == 1))), axis=0)
    if nearest_pixel[1] == 0:
        return img[:farest_pixel[0], :farest_pixel[1]]
    else:
        return img[nearest_pixel[0]:farest_pixel[0], nearest_pixel[1]:farest_pixel[1]]


def main():
    csv_file = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\data\\csv\\INbreast.csv'
    root_dir = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\data\\dicom'
    output_dir = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\data\\png'

    size = 224

    preprocess_and_save_images(csv_file, root_dir, output_dir, size)


if __name__ == "__main__":
    main()
