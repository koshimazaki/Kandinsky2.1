# Kandinsky2.1-notebooks
google collab docs to generate images using Kandinsky2.1 model


Using this code to create permutations of images and to save json files. 
```

import os
import itertools
import json
from kandinsky2 import get_kandinsky2
from PIL import Image

model = get_kandinsky2('cuda', task_type='text2img', model_version='2.1', use_flash_attention=False)

images_path_characters = '/content/drive/MyDrive/AI/characters'
images_path_environments = '/content/drive/MyDrive/AI/environments'

characters = os.listdir(images_path_characters)
environments = os.listdir(images_path_environments)

characters = [os.path.join(images_path_characters, img) for img in characters if img.endswith('.png')]
environments = [os.path.join(images_path_environments, img) for img in environments if img.endswith('.png')]

output_folder = "/content/drive/MyDrive/AI/Kandinsky 2.1/mix3"

for character_path, environment_path in itertools.product(characters, environments):
    character = Image.open(character_path)
    environment = Image.open(environment_path)
    
    images_texts = [character, environment]
    weights = [0.5, 0.5]
    images = model.mix_images(
        images_texts, 
        weights, 
        num_steps=150,
        batch_size=1, 
        guidance_scale=5,
        h=1024, w=1024,
        sampler='p_sampler', 
        prior_cf_scale=4,
        prior_steps="5"
    )
    
    character_name = os.path.basename(character_path).split('.')[0]
    environment_name = os.path.basename(environment_path).split('.')[0]
    
    for idx, image in enumerate(images):
        save_path = f"{output_folder}/K2.1_character-{character_name}_environment-{environment_name}_{idx + 1}.png"
        image.save(save_path)

        # Generate JSON file for each permutation
        json_data = {
            "character": character_name,
            "environment": environment_name
        }
        json_save_path = f"{output_folder}/metadata_{character_name}_{environment_name}_{idx + 1}.json"

        with open(json_save_path, 'w') as json_file:
            json.dump(json_data, json_file)

            torch.cuda.empty_cache()
