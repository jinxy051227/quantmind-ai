# QuantMind AI：GitHub 公开发布步骤

该仓库已经排除了本地行情快照、虚拟环境和密钥文件。**不要将个人研究 ZIP、Yahoo Finance 原始数据、API Key 或 `.env` 添加到公开仓库。**

## 方式 A：推荐，GitHub CLI

在 Mac 终端中，进入**包含 `README.md` 的项目目录**：

```bash
cd ~/Downloads/quantmind-ai
ls README.md .gitignore
```

确认 `git` 和 `gh` 可用：

```bash
git --version
gh --version
```

如缺少 `gh`，安装（需要 Homebrew）：`brew install gh`。然后登录：

```bash
gh auth login
```

选择 GitHub.com 和 HTTPS，按照终端指引在浏览器授权。不要把任何 token 发给别人。

初始化、本地检查、提交并创建**公开仓库**：

```bash
git init

git branch -M main

git add .

git status --short

git commit -m "Release QuantMind AI v1.0 research portfolio"

gh repo create quantmind-ai --public --source=. --remote=origin --push
```

`git status --short` 应只列出源代码、文档、测试及配置文件。若出现本地数据或密钥，**停止提交并先处理**。公开仓库创建后，可运行：

```bash
gh repo view --web
```

## 方式 B：不安装 `gh`

先按上面指令执行到 `git commit`。在 https://github.com/new 新建一个名称为 `quantmind-ai` 的 **Public** 空仓库。不要勾选自动初始化 README / .gitignore / License（这些文件已在本地）。创建仓库后 GitHub 会显示远端 URL：

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/quantmind-ai.git

git push -u origin main
```

将 `YOUR_GITHUB_USERNAME` 换成你实际的 GitHub 用户名。GitHub HTTPS 推送一般需要浏览器授权或凭据管理，不能直接用 GitHub 密码登录。

## 发布后检查

1. 仓库里 README 的图片/链接是否可以打开，LICENSE 是否显示 MIT。
2. Actions 的 Python 3.11、3.12、3.13 工作流是否为绿色；如果不通过先检查日志。
3. 仓库文件列表中不得存在 `.env`、`data/`、`prices_snapshot.csv`、`paper_forecasts.jsonl` 或私人研究 ZIP。
4. 在仓库右侧的 About 中填写：`Reproducible ML research for stock prediction, backtesting and risk analysis. Not investment advice.`
5. Topics 可填写 `machine-learning`, `time-series`, `quantitative-finance`, `streamlit`, `backtesting`, `research`。

## 重要说明

公开发布 v1.0 代表研究工具具备演示功能；不意味着模型能够持续盈利。请保留 AAPL 亏损结果和研究限制，不要将其描述成可用于真实资金交易的系统。
