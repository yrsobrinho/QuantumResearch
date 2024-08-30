import os
import pydicom
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
from skimage import filters
from sklearn.model_selection import train_test_split
import skimage
from skimage.color import rgb2gray


def preprocess_and_save_images(csv_file, root_dir, output_dir, size=224, test_size=0.2):
    print("Lendo o arquivo CSV...")
    data = pd.read_csv(csv_file, sep=';')
    print(f"Total de entradas no CSV: {len(data)}")

    data = data[data['Bi-Rads'].notnull()]
    print(f"Total de entradas após remover Bi-Rads nulos: {len(data)}")

    data['Label'] = data['Bi-Rads'].apply(lambda x: 0 if x in ['1', '2'] else 1)
    print("Labels atribuídos com base no Bi-Rads.")

    data['File Name'] = data['File Name'].astype(str)

    print("Dividindo os dados em conjuntos de treino e teste...")
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=data['Label'], random_state=42)
    print(f"Total de imagens no treino: {len(train_data)}, no teste: {len(test_data)}")

    train_labels = dict(zip(train_data['File Name'], train_data['Label']))
    test_labels = dict(zip(test_data['File Name'], test_data['Label']))

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
            img, img_shape = read_dicom_img(dicom_path)

            proc_img = Image_procescing_mias(img)

            resized_img = cv2.resize(proc_img, (size, size))

            label = labels_dict[file_name]
            print(f"Rótulo para {file_name}: {label}")

            label_dir = os.path.join(split_output_dir, str(label))
            output_path = os.path.join(label_dir, f"{file_name}.png")
            cv2.imwrite(output_path, resized_img * 255)  # Multiplicando por 255 para garantir que a imagem esteja em escala de 0-255
            print(f"Imagem salva em: {output_path}")

    process_and_save(train_data, 'train', train_labels)
    process_and_save(test_data, 'test', test_labels)


def get_mask_of_largest_connected_component(img_mask):
    mask, num_labels = scipy.ndimage.label(img_mask)
    mask_pixels_dict = {i: np.sum(mask == i) for i in range(1, num_labels + 1)}
    largest_mask_index = max(mask_pixels_dict, key=mask_pixels_dict.get)
    largest_mask = (mask == largest_mask_index)
    return largest_mask


def read_dicom_img(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    image = cv2.convertScaleAbs(img - np.min(img), alpha=(255.0 / min(np.max(img) - np.min(img), 10000)))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        image = np.invert(image)

    image = image / 255.0

    return image, image.shape


def Image_procescing_mias(img_mias):
    img_mias = skimage.exposure.equalize_adapthist(img_mias, clip_limit=0.03)
    threshold = filters.threshold_isodata(img_mias)
    bin_mias = (img_mias > threshold) * 1
    kernel = np.ones((5, 5), np.uint8)
    bin_mias = bin_mias.astype('uint8')
    bin_mias = cv2.erode(bin_mias, kernel, iterations=-2)

    img_mask_mias = get_mask_of_largest_connected_component(bin_mias)

    farest_pixel = np.max(list(zip(*np.where(img_mask_mias == 1))), axis=0)
    nearest_pixel = np.min(list(zip(*np.where(img_mask_mias == 1))), axis=0)

    if nearest_pixel[1] == 0:
        croped = img_mias[:farest_pixel[0], :farest_pixel[1]]
    else:
        croped = img_mias[nearest_pixel[0]:, nearest_pixel[1]:farest_pixel[1]]

    return croped


def main():
    csv_file = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\INbreast\\data\\csv\\INbreast.csv'
    root_dir = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\INbreast\\data\\dicom'
    output_dir = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\INbreast\\data\\png'
    size = 224

    preprocess_and_save_images(csv_file, root_dir, output_dir, size)


if __name__ == "__main__":
    main()
