#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像搜索脚本
根据文本查询在数据库中搜索相似的图像
"""

import torch
import numpy as np
import psycopg2
import os

# 尝试导入Chinese-CLIP模型
try:
    from cn_clip import clip
    from cn_clip.clip import load_from_name, tokenize
    CN_CLIP_AVAILABLE = True
except ImportError:
    print("警告: 未找到Chinese-CLIP库，将无法使用文本特征提取功能")
    CN_CLIP_AVAILABLE = False

# 设备设置 - 添加对MPS设备的支持
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("检测到CUDA设备")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("检测到MPS设备")
    else:
        device = "cpu"
        print("使用CPU设备")
    return device

device = get_device() if CN_CLIP_AVAILABLE else "cpu"
print(f"使用设备: {device}")

# PostgreSQL连接配置
DB_CONFIG = {
    'host': 'localhost',
    'database': 'image_search',
    'user': os.environ.get('PG_USER', 'postgres'),  # 从环境变量获取用户名，如果没有则默认为postgres
    'password': os.environ.get('PG_PASSWORD', 'postgres'),  # 从环境变量获取密码
    'port': 5432
}

def get_db_connection():
    """获取数据库连接"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"数据库连接失败: {e}")
        print("请确保：")
        print("1. PostgreSQL服务正在运行")
        print("2. 数据库配置正确（可以在环境变量中设置PG_USER和PG_PASSWORD）")
        print("3. 用户名和密码正确")
        raise

def load_clip_model():
    """加载CLIP模型"""
    if not CN_CLIP_AVAILABLE:
        print("错误: Chinese-CLIP库不可用")
        return None, None
        
    global model, preprocess
    if 'model' not in globals():
        model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./models')
        model.eval()
    return model, preprocess

def embed_text(text):
    """将文本转换为向量"""
    if not CN_CLIP_AVAILABLE:
        print("错误: Chinese-CLIP库不可用")
        return None
        
    model, _ = load_clip_model()
    
    try:
        # 对文本进行编码
        text_tokens = tokenize([text]).to(device)
        
        # 提取特征
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            # 归一化
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            vector = text_features.squeeze().cpu().numpy()
            
        return vector
    except Exception as e:
        print(f"处理文本 '{text}' 时出错: {str(e)}")
        return None

def search_images_by_text(text, limit=5):
    """使用文本检索相似的图像"""
    try:
        # 将文本转换为向量
        text_vector = embed_text(text)
        if text_vector is None:
            return []
        
        # 查询数据库
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 使用内积相似度搜索最相似的图像
        # 将Python列表转换为PostgreSQL数组格式，并明确指定为vector类型
        vector_array = text_vector.tolist()
        cur.execute("""
            SELECT image_name, image_path, 1 - (embedding <=> %s::vector) AS similarity
            FROM images
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_array, vector_array, limit))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # 格式化结果
        formatted_results = []
        for row in results:
            formatted_results.append({
                "image_name": row[0],
                "image_path": row[1],
                "similarity": row[2]
            })
        
        print(f"与 '{text}' 最相关的 {len(formatted_results)} 张图像:")
        for i, result in enumerate(formatted_results):
            print(f"  {i+1}. 图像: {result['image_name']}")
            print(f"     路径: {result['image_path']}")
            print(f"     相似度: {result['similarity']:.4f}")
            print()
        
        return formatted_results
    except Exception as e:
        print(f"搜索图像时出错: {e}")
        error_msg = str(e)
        if "permission denied to create extension" in error_msg:
            print("提示：请使用PostgreSQL超级用户手动创建pgvector扩展:")
            print("  1. 连接到数据库: psql -U postgres -d image_search")
            print("  2. 执行命令: CREATE EXTENSION IF NOT EXISTS vector;")
        elif "type \"vector\" does not exist" in error_msg:
            print("提示：vector类型不存在，请确保:")
            print("  1. pgvector扩展已正确安装")
            print("  2. 已使用超级用户在当前数据库中创建了vector扩展")
            print("  3. 执行命令: CREATE EXTENSION IF NOT EXISTS vector;")
        elif "operator does not exist" in error_msg:
            print("提示：vector操作符不存在，请确保:")
            print("  1. pgvector扩展已正确安装并启用")
            print("  2. 数据库中已正确创建vector类型")
            print("  3. 查询中正确使用了vector类型转换")
        return []

if __name__ == "__main__":
    import sys
    
    # 加载模型
    load_clip_model()
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python search_images.py <查询文本> [结果数量]")
        sys.exit(1)
    
    query_text = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # 执行搜索
    search_images_by_text(query_text, limit)