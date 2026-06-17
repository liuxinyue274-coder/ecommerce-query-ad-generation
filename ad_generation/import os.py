import os
import torch
import clip
from PIL import Image
import pickle
from tqdm import tqdm

# ================== 配置区 ==================
image_dir = "./image"
sketch_dir = "./sketch"
output_file = "match_results.txt"
feature_cache = "sketch_features.pkl"  # 缓存特征的文件
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 加载模型 ==================
model, preprocess = clip.load("ViT-B/32", device=device)

# ================== 工具函数 ==================
def get_image_feature(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encode_image(image)
    return feature / feature.norm(dim=-1, keepdim=True)

def compute_similarity(feat1, feat2):
    return (feat1 @ feat2.T).item()

def precompute_sketch_features(sketch_dir):
    """预计算所有草图的特征并缓存，避免重复计算"""
    # 尝试加载缓存的特征
    if os.path.exists(feature_cache):
        print(f"Loading cached sketch features from {feature_cache}")
        with open(feature_cache, 'rb') as f:
            return pickle.load(f)

    print("Precomputing sketch features...")
    sketch_features = {}
    sketch_files = [f for f in os.listdir(sketch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for sketch_file in tqdm(sketch_files, desc="Encoding sketches"):
        sketch_path = os.path.join(sketch_dir, sketch_file)
        feature = get_image_feature(sketch_path)
        sketch_features[sketch_file] = feature

    # 保存到缓存文件
    with open(feature_cache, 'wb') as f:
        pickle.dump(sketch_features, f)
    print(f"Saved sketch features to {feature_cache}")
    return sketch_features

# ================== 主程序 ==================
if __name__ == "__main__":
    # 预计算所有草图的特征向量
    sketch_features = precompute_sketch_features(sketch_dir)

    # 获取所有图像文件并按名称排序
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    results = []

    for img_file in tqdm(images, desc="Matching images to sketches"):
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)

        # 根据命名规则，从预计算的特征字典中筛选出与该图像相关的草图特征
        # 修改下面的筛选规则以匹配你的实际命名逻辑
        related_sketches = {name: feat for name, feat in sketch_features.items() if name.startswith(img_name)}

        if not related_sketches:
            print(f"Warning: No sketches found for {img_file}, skipping.")
            results.append(f"{img_file}\tNO MATCH FOUND\t0.0\n")
            continue

        # 提取图像特征
        img_feature = get_image_feature(img_path)

        # 在相关的草图中寻找最佳匹配
        best_sketch_name = None
        best_score = -1
        for sketch_name, sketch_feat in related_sketches.items():
            score = compute_similarity(img_feature, sketch_feat)
            if score > best_score:
                best_score = score
                best_sketch_name = sketch_name

        results.append(f"{img_file}\t{best_sketch_name}\t{best_score:.4f}\n")

    # 保存结果
    with open(output_file, 'w') as f:
        f.write("# Image\tMatched Sketch\tSimilarity Score\n")
        f.writelines(results)

    print(f"Finished! Results saved to {output_file}")