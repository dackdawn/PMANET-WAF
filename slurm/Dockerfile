# 这个是基于Slurm集群的环境

# 基于 continuumio/anaconda3:main 镜像，已包含 Conda 和 Python (Anaconda)
FROM continuumio/anaconda3:main

# 切换到 bash shell (可选，更方便交互式操作)
SHELL ["/bin/bash", "-c"]

# 设置 Conda base 环境的 Python 版本为 3.8 (如果基础镜像的 base 环境不是 Python 3.8)
RUN conda install -n base python=3.8 -y

# 激活 base 环境 (确保后续操作在 base 环境中进行)
ENV CONDA_DEFAULT_ENV=base
ENV CONDA_ENV_PATH=/opt/anaconda
ENV PATH="$CONDA_ENV_PATH/bin:${PATH}"

# 安装 PyTorch (GPU 版本, CUDA 11.8，请根据实际情况修改)
RUN conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装 依赖
RUN pip install scikit-learn matplotlib seaborn pytorch-pretrained-bert transformers==4.46.3

# 清理 conda 缓存，减小镜像大小
RUN conda clean -p -t -y

# 设置工作目录 (可选)
# WORKDIR /app

# 定义启动命令 (可选，根据你的实际应用场景设置)
# CMD ["/bin/bash"]