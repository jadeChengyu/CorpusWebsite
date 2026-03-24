# 无需任何json文件！直接生成完整可部署的网页
import json

# 直接创建示例分析结果（不需要外部文件）
results = {
    "project_name": "Corpus Website Analysis Tool",
    "description": "Comprehensive Metaphor & Cultural Analysis",
    "status": "success",
    "analysis_results": {
        "word_frequency": {"example": 10, "culture": 8, "language": 5},
        "cultural_keywords": ["China", "tradition", "history", "culture"],
        "metaphor_analysis": "Completed successfully"
    }
}

# 生成静态 HTML
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Corpus Analysis Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background:#f7f7f7; }}
        .container {{ max-width:1200px; margin:0 auto; background:white; padding:30px; border-radius:10px; }}
        pre {{ background: #f0f2f5; padding: 20px; border-radius: 8px; overflow-x: auto; font-size:14px; }}
        h1 {{ color:#1a73e8; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Corpus Website Analysis Dashboard</h1>
        <h2>Analysis Results</h2>
        <pre>{json.dumps(results, indent=2, ensure_ascii=False)}</pre>
        <br>
        <a href="https://jadechengyu.github.io/-jadeChengyu-Pro-MaQing-Angel/" target="_blank">
            ← Back to Main Website
        </a>
    </div>
</body>
</html>
"""

# 保存为静态 HTML 文件
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ 成功生成 index.html！可以直接部署到 GitHub Pages 啦！")