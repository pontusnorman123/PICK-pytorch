from datasets import load_dataset

# this dataset uses the new Image feature :)
#dataset = load_dataset("Hyeoli/layoutlmv3_cord")
#dataset = load_dataset("pontusnorman123/swe_set2")
dataset_dict = load_dataset("pontusnorman123/swe_set2")

import pandas as pd
from PIL import Image
import os


# Mapping dictionary for ner_tags
tag_mapping = {
    0: 'COMPANY',
    1: 'DATE',
    2: 'ADDRESS',
    3: 'TOTAL',
    4: 'others'
}


# Function to process dataset and save data in specified folders
def process_dataset(dataset, dataset_type, base_path):
    # Create directories for outputs specific to dataset type
    os.makedirs(f"{base_path}/boxes_and_transcripts", exist_ok=True)
    os.makedirs(f"{base_path}/images", exist_ok=True)

    samples_list = []

    for i, item in enumerate(dataset):
        file_name = f"{item['id']}.tsv"
        image_file_name = f"{item['id']}.jpg"
        samples_list.append([i, dataset_type, image_file_name])

        # Save image to dataset-specific images folder
        image_object = item['image']
        if hasattr(image_object, 'save'):
            image_object.save(f"{base_path}/images/{image_file_name}")
        else:
            print(
                f"Skipping image save for ID {item['id']} in {dataset_type} because the image object has no 'save' method.")

        # Write boxes and transcripts to a dataset-specific .tsv file
        with open(f"{base_path}/boxes_and_transcripts/{file_name}", 'w') as f:
            for word, bbox, tag in zip(item['words'], item['bboxes'], item['ner_tags']):
                # Convert bbox from [x_min, y_min, x_max, y_max] to 8 values (clockwise)
                if len(bbox) == 4:
                    x_min, y_min, x_max, y_max = bbox
                    bbox_str = ','.join(map(str, [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]))
                elif len(bbox) == 8:
                    bbox_str = ','.join(map(str, bbox))
                else:
                    print(f"Unexpected bbox format for ID {item['id']}: {bbox}")
                    continue

                # Map the numerical tag to a label
                tag_label = tag_mapping.get(tag, 'OTHERS')
                f.write(f"{i},{bbox_str},{word},{tag_label}\n")

    # Create samples_list.csv for both training and testing data
    df = pd.DataFrame(samples_list, columns=['index', 'document_type', 'file_name'])
    df.to_csv(f"{base_path}/{dataset_type}_samples_list.csv", index=False)


# Paths for the dataset types
train_base_path = 'data/data_examples_root'
test_base_path = 'data/test_data_example'

# Process training and testing datasets
process_dataset(dataset_dict['train'], 'train', train_base_path)
process_dataset(dataset_dict['test'], 'test', test_base_path)
