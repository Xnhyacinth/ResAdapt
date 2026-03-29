import requests
import base64
import io
from PIL import Image

# 服务端地址 (使用 HTTP 端口 8000)
SERVER_URL = "http://22.0.156.69:8000/"

# 1. 辅助函数：将图片转为 Base64 字符串
def encode_image(image_path=None, pil_image=None):
    img = pil_image
    if image_path:
        img = Image.open(image_path)
    
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG") # 或 PNG
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# 2. 准备数据
print("🖼️ Preparing request data...")
dummy_img = Image.new('RGB', (224, 224), color='red')
image_b64 = encode_image(pil_image=dummy_img)

payload = {
    "messages": [{"role": "user", "content": "<|image_1|>\nDescribe this image."}],
    "visuals": [image_b64]
}

# 🟢 关键修改：定义 Headers
# 这里的 value 通常有几种可能，建议按顺序尝试：
# 1. "default" (默认 Ray Serve 应用名)
# 2. "DynamicPredictorService" (你的类名/部署名)
# 3. "app" (你代码里的变量名)
# 4. 如果是在内部平台部署，可能是平台分配的 Service ID
headers = {
    "Content-Type": "application/json",
    # 🟢 加上网关要求的 Header
    "Destination-Service": "merlin.mlx.designer" 
}

print(f"🚀 Sending request to {SERVER_URL} with headers...")

try:
    # 🟢 在 post 请求中加入 headers=headers
    response = requests.post(SERVER_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("✅ Prediction Result:", response.json())
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        print("尝试修改 header value，或者咨询网络管理员该网关要求的服务名是什么。")

except Exception as e:
    print(f"❌ Connection failed: {str(e)}")








# import ray
# from ray import serve
# import torch
# from PIL import Image

# # 1. 连接到 Ray Serve
# # 确保你已经执行了 serve run dynamic_serve:app
# ray.init(address="ray://127.0.0.1:10001", namespace="serve")
# handle = serve.get_app_handle("default")

# # ==========================================
# # 步骤 1: 初始化模型 (调用 initialize_model)
# # ==========================================
# print("\n--- [Step 1] Initializing Model ---")
# model_path = "/mnt/bn/jiangzhongtao/users/liaohuanxuan/models/predictor_flash"

# # 注意：调用自定义方法要用 handle.方法名.remote()
# init_ref = handle.initialize_model.remote(model_path)
# print(init_ref.result())


# # ==========================================
# # 步骤 2: 执行推理 (调用 __call__)
# # ==========================================
# print("\n--- [Step 2] Running Inference ---")

# # 构造你的复杂数据 (建议在 Client 端用 CPU Tensor)
# dummy_img = Image.new('RGB', (100, 100))
# inputs = {
#     'input_ids': torch.randint(0, 100, (2, 10)),
#     'attention_mask': torch.ones((2, 10)),
#     'pixel_values': torch.randn((2, 3, 224, 224)),
#     'image_grid_thw': torch.tensor([[1, 22, 34], [1, 22, 34]]),
#     'multi_modal_data': [{'image': [dummy_img]}, {'image': [dummy_img]}],
#     'text': ['prompt 1', 'prompt 2']
# }

# # 直接调用 handle.remote() 就会触发服务端的 __call__
# predict_ref = handle.remote(inputs)
# result = predict_ref.result()
# print(f"Prediction Result: {result}")


# # ==========================================
# # 步骤 3: 删除模型 (调用 release_model)
# # ==========================================
# print("\n--- [Step 3] Releasing Model ---")
# release_ref = handle.release_model.remote()
# print(release_ref.result())


# # ==========================================
# # 步骤 4: 验证已删除
# # ==========================================
# print("\n--- [Step 4] Verify Deletion ---")
# # 再次调用推理，应该报错提示未初始化
# verify_ref = handle.remote(inputs)
# print(f"Result after release: {verify_ref.result()}")