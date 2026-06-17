# Deployment Guide

## GitHub

初始化并提交：

```bash
git init
git add .
git commit -m "Prepare project for GitHub and demo deployment"
```

创建远程仓库后，绑定并推送：

```bash
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

推送前确认不要提交：

- 完整 KuaiSearch 数据目录
- API Key、`.env`、`secrets.toml`
- 模型权重、checkpoint、缓存文件
- 大体积 JSONL/CSV/PKL/PT/BIN/SafeTensors 输出

## Streamlit Community Cloud

推荐部署 `demo_app/app.py`。

1. 将仓库推送到 GitHub。
2. 打开 Streamlit Community Cloud。
3. 选择该 GitHub 仓库和目标分支。
4. Main file path 填：

```text
demo_app/app.py
```

5. Python dependencies 使用：

```text
demo_app/requirements.txt
```

6. 如果使用 `deepseek_chat`，在 Streamlit 的 Secrets 或环境变量里配置：

```toml
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"
```

不要把 API Key 写进代码、README、脚本或提交记录。

默认展示页面读取：

```text
demo_app/outputs/demo_v5_outputs_sample.jsonl
```

该样例文件适合公开部署。完整实时链路依赖本地 KuaiSearch lite 数据，云端部署时通常只建议展示离线样例。

## Vercel

Vercel 不建议直接部署 Streamlit 应用。Streamlit 是 Python 长进程应用，更适合 Streamlit Community Cloud、Hugging Face Spaces、Render 或自有服务器。

Vercel 只建议用于：

- 部署静态 `demo_app/public_snapshot/`
- 部署项目介绍页
- 部署文档或静态 HTML 版本

如果使用 Vercel 静态站点，入口目录可选择：

```text
demo_app/public_snapshot
```

如果后续需要在线 API，请通过环境变量或服务端 Secrets 管理密钥，不要在前端代码中暴露。
