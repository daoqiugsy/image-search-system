#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
图像插入脚本
将图像文件转换为向量并插入到PostgreSQL数据库中
"""

import torch
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import os
import glob
from PIL import Image

# 尝试导入Chinese-CLIP模型
try:
    from cn_clip import clip
    from cn_clip.clip import load_from_name, tokenize, image_transform
    CN_CLIP_AVAILABLE = True
except ImportError:
    print("警告: 未找到Chinese-CLIP库，将无法使用图像特征提取功能")
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
    'user': os.environ.get('PG_USER', 'postgres'),
    'password': os.environ.get('PG_PASSWORD', 'postgres'),
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

def init_database():
    """初始化数据库和表"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 创建图像表（如果不存在）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id SERIAL PRIMARY KEY,
                image_name VARCHAR(255) UNIQUE,
                image_path TEXT,
                embedding VECTOR(512)
            );
        """)
        
        # 显式创建唯一索引以支持ON CONFLICT子句
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_images_image_name 
            ON images (image_name);
        """)
        
        # 创建向量索引（如果不存在）
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_images_embedding 
            ON images 
            USING hnsw (embedding vector_ip_ops);
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        print("数据库初始化完成")
        return True
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False

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

def embed_image(image_path):
    """将图像转换为向量"""
    if not CN_CLIP_AVAILABLE:
        print("错误: Chinese-CLIP库不可用")
        return None
        
    model, preprocess = load_clip_model()
    
    try:
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return None
            
        # 打开图像并预处理
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # 提取特征
        with torch.no_grad():
            image_features = model.encode_image(image)
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # 转换为numpy数组
            vector = image_features.squeeze().cpu().numpy()
            
        return vector
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return None

def insert_single_image(image_path):
    """插入单张图像到数据库"""
    try:
        # 获取图像向量
        vector = embed_image(image_path)
        if vector is None:
            return False
            
        # 获取图像名称
        image_name = os.path.basename(image_path)
        image_abs_path = os.path.abspath(image_path)
        
        # 插入数据库
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 使用ON CONFLICT处理重复的image_name
        cur.execute("""
            INSERT INTO images (image_name, image_path, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (image_name) 
            DO UPDATE SET 
                image_path = EXCLUDED.image_path,
                embedding = EXCLUDED.embedding
        """, (image_name, image_abs_path, vector.tolist()))
        
        conn.commit()
        cur.close()
        conn.close()
        
        print(f"图像 {image_name} 已插入或更新数据库")
        return True
    except Exception as e:
        print(f"插入图像时出错: {e}")
        return False

def batch_insert_images(image_dir, extensions=["*.jpg", "*.jpeg", "*.png", "*.bmp"]):
    """批量插入指定目录下的所有图像"""
    # 获取所有图像路径
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    
    if not image_paths:
        print(f"在目录 '{image_dir}' 中未找到图像文件")
        return 0
    
    print(f"找到 {len(image_paths)} 张图像，开始批量插入...")
    
    # 加载模型
    load_clip_model()
    
    # 批量处理图像
    image_data_list = []
    for image_path in image_paths:
        vector = embed_image(image_path)
        if vector is not None:
            image_name = os.path.basename(image_path)
            image_abs_path = os.path.abspath(image_path)
            image_data_list.append({
                'image_name': image_name,
                'image_path': image_abs_path,
                'vector': vector
            })
    
    # 批量插入数据库
    inserted_count = 0
    if image_data_list:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # 使用ON CONFLICT处理重复的image_name
            execute_values(
                cur,
                """
                INSERT INTO images (image_name, image_path, embedding) VALUES %s
                ON CONFLICT (image_name) 
                DO UPDATE SET 
                    image_path = EXCLUDED.image_path,
                    embedding = EXCLUDED.embedding
                """,
                [(data['image_name'], data['image_path'], data['vector'].tolist()) for data in image_data_list]
            )
            
            conn.commit()
            cur.close()
            conn.close()
            inserted_count = len(image_data_list)
            print(f"批量处理 {inserted_count} 条记录完成（插入或更新）")
        except Exception as e:
            print(f"批量插入时出错: {e}")
    
    return inserted_count

if __name__ == "__main__":
    import sys
    
    # 初始化数据库
    if not init_database():
        print("数据库初始化失败，请检查数据库配置")
        sys.exit(1)
    
    # 加载模型
    load_clip_model()
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法:")
        print("  插入单张图像: python insert_images.py /path/to/image.jpg")
        print("  批量插入图像: python insert_images.py /path/to/image_directory")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # 判断是单个文件还是目录
    if os.path.isfile(path):
        # 插入单张图像
        insert_single_image(path)
    elif os.path.isdir(path):
        # 批量插入图像
        batch_insert_images(path)
    else:
        print(f"路径 '{path}' 不存在")
        sys.exit(1)