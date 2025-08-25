# 图像搜索系统

一个基于Chinese-CLIP的跨模态图像搜索系统，支持通过文本查询搜索相似图像。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 功能特性

1. 使用Chinese-CLIP模型提取图像和文本特征
2. 基于PostgreSQL和pgvector实现向量存储和检索
3. 支持单张图像插入和批量图像插入
4. 支持基于文本的图像检索
5. 自动检测并使用最佳计算设备（CUDA/MPS/CPU）

## 许可证

本项目采用 MIT 许可证，详情请参见 [LICENSE](LICENSE) 文件。

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

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目。

## 致谢

- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) - 本项目使用的跨模态模型
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL的向量扩展