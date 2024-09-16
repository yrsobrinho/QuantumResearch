import os


def rename_dicom_files(root_dir):
    dicom_files = [f for f in os.listdir(root_dir) if f.endswith('.dcm')]

    for dicom_file in dicom_files:
        old_path = os.path.join(root_dir, dicom_file)

        # Extrai a parte do nome antes do primeiro '_'
        new_file_name = dicom_file.split('_')[0] + '.dcm'
        new_path = os.path.join(root_dir, new_file_name)

        # Renomeia o arquivo, se o novo nome não for igual ao antigo
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"Renomeado: {dicom_file} -> {new_file_name}")
        else:
            print(f"O arquivo {dicom_file} já está renomeado corretamente.")


if __name__ == "__main__":
    root_dir = 'C:\\Users\\Win10\\PycharmProjects\\QuantumResearch\\data\\dicom'
    rename_dicom_files(root_dir)
