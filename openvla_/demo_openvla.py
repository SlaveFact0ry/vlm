from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import datasets
import io
from matplotlib import pyplot as plt
import numpy as np
# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")


ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "nyu_door_opening_surprising_effectiveness", trust_remote_code=True)#, streaming=True, split='train')  # IterDataset

samples = ds['train'][1]['data.pickle']['steps'] # 25 steps, action/observation 등


images = []
actions = []
frame_indices = []
frame_idx = 0

for sample in samples:
    encoded_image = sample['observation']['image']['bytes'] # 
    bytes_str = sample['observation']['natural_language_instruction'] # 
    INSTRUCTION = bytes_str.decode("utf-8")  # UTF-8 디코딩

    # 바이트 데이터를 이미지로 변환
    image = Image.open(io.BytesIO(encoded_image))
    images.append(image)

    # action

    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"


    # Predict Action (7-DoF; un-normalize for BridgeV2)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    actions.append(action)

    frame_indices.append(frame_idx)
    frame_idx += 1



images[0].save(f"/home/baekw92/practice_vla/output_1.gif", save_all=True, append_images=images[1:], duration=500, loop=0)

actions = np.array(actions)

# Plot actions over time
plt.figure(figsize=(12, 6))

# 7개 요소를 각각 그래프로 표현
action_labels = ["Δx", "Δy", "Δz", "Roll", "Pitch", "Yaw", "Grip"]
for i, label in enumerate(action_labels):
    plt.plot(frame_indices, actions[:, i], label=label)

plt.xlabel("Frame Index")
plt.ylabel("Action Value")
plt.legend()
plt.title("Predicted Robot Actions Over Time")
plt.savefig(f"/home/baekw92/practice_vla/output_actions_1.png")
plt.close()



