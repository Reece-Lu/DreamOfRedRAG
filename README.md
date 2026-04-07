# 红楼梦 RAG 项目

## 结构

```text
Dream_of_the_Red_Chamber_RAG/
├─ data/
│  └─ chroma_db/
├─ src/
│  └─ rag_app/
│     ├─ config.py
│     ├─ core.py
│     └─ cli.py
├─ rag_minimal.py
├─ .env
├─ .env.example
├─ .gitignore
├─ Dockerfile
├─ docker-compose.yml
└─ requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
```

## 配置密钥

复制 `.env.example` 为 `.env`，然后填写你的 OpenAI API Key。

```bash
copy .env.example .env
```

PowerShell 也可以用：

```powershell
Copy-Item .env.example .env
```

不要把真实密钥提交到 Git。

## 本地运行

导入语料：

```bash
python rag_minimal.py ingest
```

只检索，不调用大模型：

```bash
python rag_minimal.py search "贾宝玉第一次见到林黛玉在哪里？"
```

提问并生成答案：

```bash
python rag_minimal.py ask "贾宝玉第一次见到林黛玉在哪里？"
```

## Docker 运行

构建镜像：

```bash
docker compose build
```

导入语料：

```bash
docker compose run --rm rag python rag_minimal.py ingest
```

提问：

```bash
docker compose run --rm rag python rag_minimal.py ask "贾宝玉第一次见到林黛玉在哪里？"
```

这个版本把 RAG 拆成了四层：文本切块、OpenAI embeddings、向量检索、LLM 回答、命令行入口。`OPENAI_API_KEY` 只放在 `.env` 里，不进仓库。

如果检索结果不准，通常先尝试：

- 重新执行 `ingest`
- 把 `ask` 的 `--top-k` 调大一点
- 把切块调细一点，或者继续按章节拆分语料

如果你之前已经用本地 embedding 建过 `chroma_db`，请先删除旧库再重新 `ingest`，因为现在换成了 OpenAI embedding，旧向量不能直接共用。
