FROM sais-public-registry.cn-shanghai.cr.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04

ADD . /app

# 设置工作目录
WORKDIR /app

# 设置pip国内源，尝试不同的源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 逐个安装依赖包，以便找出冲突的包
COPY requirements.txt .
RUN cat requirements.txt | xargs -n 1 pip install

# 定义容器启动时默认执行的命令
CMD ["sh", "/app/run.sh"]