import datasets
import io
from PIL import Image
from matplotlib import pyplot as plt

# ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "berkeley_cable_routing", trust_remote_code=True)
ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "nyu_door_opening_surprising_effectiveness", trust_remote_code=True)

print(ds)


samples = ds['train'][0]['data.pickle']['steps'] # 25 steps, action/observation 등


images = []
for sample in samples:
    encoded_image = sample['observation']['image']['bytes'] # 
    instruct = sample['observation']['natural_language_instruction'] # 
    print(instruct)

    # 바이트 데이터를 이미지로 변환
    image = Image.open(io.BytesIO(encoded_image))
    images.append(image)
    
    

images[0].save(f"/home/baekw92/practice_vla/output.gif", save_all=True, append_images=images[1:], duration=500, loop=0)

print('finish')