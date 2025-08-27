# 多模态文搜图系统

一个基于Chinese-CLIP的跨模态图像搜索系统，支持通过文本查询搜索相似图像。

## 核心原理

本系统基于跨模态检索技术，通过统一语义空间实现文本与图像的关联匹配，核心原理如下：

1. **特征提取**：
   - 使用Chinese-CLIP模型将图像和文本分别编码为512维向量
   - 文本编码示例：`text_features = model.encode_text(text_input)`
   - 图像编码示例：`image_features = model.encode_image(image_input)`

2. **向量存储**：
   - 采用PostgreSQL的VECTOR数据类型存储特征向量
   - 自动构建HNSW索引加速近似最近邻搜索
   - 数据表结构示例：
     ```sql
     CREATE TABLE image_embeddings (
         id SERIAL PRIMARY KEY,
         path TEXT NOT NULL,
         embedding VECTOR(512)
     );
     ```

3. **相似度计算**：
   - 使用余弦相似度度量文本与图像的语义匹配度
   - 检索流程：
     ```python
     # 文本特征提取
     text_features = model.encode_text(query_text)
     
     # 向量相似度查询
     results = db.query("SELECT path FROM image_embeddings ORDER BY embedding <-> %s LIMIT 10", 
                        (text_features,))
     ```

## 架构图
![img.png](img.png)

## 功能特性

1. 使用Chinese-CLIP模型提取图像和文本特征
2. 基于PostgreSQL和pgvector实现向量存储和检索
3. 支持单张图像插入和批量图像插入
4. 支持基于文本的图像检索
5. 自动检测并使用最佳计算设备（CUDA/MPS/CPU）

## 安装依赖

确保已安装PostgreSQL并创建了数据库，然后安装Python依赖：

```bash
pip install -r requirements.txt
```

同时需要安装pgvector扩展：

```bash
# 方法1: 使用PostgreSQL超级用户手动安装（推荐）
sudo -u postgres psql -d image_search -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 方法2: 如果你是数据库超级用户
psql -U postgres -d image_search -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 方法3: 在PostgreSQL命令行中执行
\c image_search
CREATE EXTENSION IF NOT EXISTS vector;

# 验证扩展是否正确安装
\dx vector
```

## 配置数据库连接

设置环境变量：

```bash
export PG_USER=your_username
export PG_PASSWORD=your_password
```

或者修改代码中的数据库配置部分。

## 使用示例

```bash
# 插入单张图像
python insert_images.py /path/to/image.jpg

# 批量插入图像
python insert_images.py /path/to/image_directory

# 根据文本搜索图像
python search_images.py "北京烤鸭" 10
```