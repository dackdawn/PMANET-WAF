docker build -t lzp-slurm-pmanet:latest .
docker save -o lzp-slurm-pmanet.tar lzp-slurm-pmanet:latest

# 传到服务器上之后

# 创建运行环境，只需要一次就好，生成lzp-slurm-pmanet.sif
singularity build lzp-slurm-pmanet.sif docker-archive://$PWD/lzp-slurm-pmanet.tar

# 运行批处理文件（方法一）
sbatch main.sbatch

# 抢占式运行（方法二）
# 1个节点，8个cpu核心
salloc -p A800 -N1 -n8 --gres=gpu:1 -q qmultiple -J Test-lzp

# 创建完后可以连接上去，看情况是哪个
ssh node33

# 查看是哪个任务
squeue

# 终止任务
scancel 23442

# 抢占式下执行命令
singularity exec --nv slurm/lzp-slurm-pmanet.sif bash