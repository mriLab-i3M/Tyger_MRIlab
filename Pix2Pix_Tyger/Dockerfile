FROM python:3.10-slim

RUN pip install numpy scipy matplotlib mrd-python torch torchvision dominate visdom wandb pytorch_ssim

WORKDIR /app
COPY scripts/ /app/

CMD ["python3"]
