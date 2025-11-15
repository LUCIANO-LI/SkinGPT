import os
import tempfile
import traceback
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS

analyzer = None


def try_import_analyzer() -> None:
    global analyzer
    if analyzer is not None:
        return
    try:
        from SuperSkinGPT import SmartSkinAnalyzer
        model_path = os.environ.get(
            "SKIN_MODEL_PATH",
            os.path.join(os.path.dirname(__file__), "models", "super_multitask_skinnet.pth"),
        )
        analyzer = SmartSkinAnalyzer(model_path=model_path)
    except Exception:
        analyzer = None


def build_fallback_response() -> Dict[str, Any]:
    return {
        "error": "模型未加载，且已禁用回退结果",
        "code": "MODEL_NOT_LOADED"
    }


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-DeepSeek-Key, Authorization"
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    return resp


@app.get("/health")
def health():
    try:
        try_import_analyzer()
        return jsonify({
            "ok": True,
            "modelLoaded": analyzer is not None,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/analyze")
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "缺少图片文件字段: image"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "未选择图片文件"}), 400

    use_tta = request.args.get("use_tta", "true").lower() == "true"

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".jpg") as tmp:
        temp_path = tmp.name
        file.save(temp_path)

    try:
        try_import_analyzer()

        if analyzer is None:
            return jsonify({
                "error": "模型未加载，请检查依赖与权重路径(环境变量 SKIN_MODEL_PATH 或 models/super_multitask_skinnet.pth)",
                "code": "MODEL_NOT_LOADED"
            }), 503

        result = analyzer.analyze_image(temp_path, use_tta=use_tta, save_results=False)
        preds = result.get("predictions", {})
        task_scores = preds.get("task_predictions")
        total_score = preds.get("total_score")

        model_task_names = getattr(analyzer, "task_names", [
            "细纹/皱纹 (Fine Lines/Wrinkles)",
            "色素沉着 (Pigmentation)",
            "毛孔 (Pores)",
            "红血丝 (Redness)",
            "紫外线损伤 (UV Damage)",
            "整体质地 (Overall Texture)",
        ])

        tasks = []
        if task_scores is not None:
            scores = [float(s) for s in list(task_scores)]
            for name, score_val in zip(model_task_names, scores):
                level = (
                    "优秀Q1" if score_val <= 0.5 else
                    "良好Q2" if score_val <= 1.5 else
                    "需要关注Q3" if score_val <= 2.5 else
                    "需要改善Q4"
                )
                tasks.append({"name": name, "score": round(score_val, 2), "level": level})

        resp = {
            "totalScore": round(float(total_score) if total_score is not None else 1.5, 2),
            "derived": {},
            "analysis": {
                "tasks": tasks,
                "suggestions": {
                    "critical": [{"task": "紫外线损伤", "advice": "加强防晒，使用修复精华"}],
                    "preventive": [{"task": "色素沉着", "advice": "避免暴晒，配合淡斑护理"}],
                    "good": [{"task": "整体质地", "advice": "保持清洁保湿与规律作息"}],
                },
            },
            "source": "model"
        }
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
import json
import re
from typing import Optional


def call_deepseek(tasks, api_key: Optional[str] = None):
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    compact = [
        {"name": t.get("name"), "score": float(t.get("score", 1.5))}
        for t in (tasks or [])
    ]

    system_msg = (
        "你是专业皮肤顾问与美妆产品推荐专家。请基于用户的六项皮肤评分(0~3，分数越高问题越严重)给出改善建议与可执行的产品推荐。"
        "务必使用简体中文，并按照JSON输出，字段: overview(50字内), actions(按优先级数组, 每项含 title, why, how), "
        "products(数组, 每项含 step(日常步骤), product_type, key_ingredients(数组), examples(2-3 个中文商品名), reason)。"
    )

    user_prompt = {
        "scores": compact,
        "scale": "0~3 (高=更需要改善)",
        "requirements": [
            "优先级从高到低排列",
            "不出现医学诊断与夸大疗效",
            "总字数控制在300字左右",
        ],
    }

    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        }

        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as he:
            raw = he.read().decode("utf-8", errors="ignore")
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or "{}"

        try:
            return json.loads(content)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                return json.loads(match.group(0))
            return {"overview": content}
    except Exception as e:
        traceback.print_exc()
        return {"_error": str(e)}


@app.route("/ai_advice", methods=["POST", "OPTIONS"])
def ai_advice():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        payload = request.get_json(force=True, silent=True) or {}
        tasks = payload.get("tasks", [])
        incoming_key = (
            request.headers.get("x-deepseek-key")
            or payload.get("apiKey")
            or (request.headers.get("Authorization") or "").replace("Bearer ", "").strip() or None
            or None
        )
        result = call_deepseek(tasks, api_key=incoming_key)
        if result is None:
            return jsonify({"error": "未配置 DeepSeek 密钥或服务不可用"}), 400
        if isinstance(result, dict) and result.get("_error"):
            return jsonify({"error": f"DeepSeek 请求失败: {result['_error']}"}), 502
        return jsonify({"aiAdvice": result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"AI建议服务异常: {str(e)}"}), 500

@app.get("/diag")
def diag():
    try:
        return jsonify({
            "apiKey": bool(os.environ.get("DEEPSEEK_API_KEY")),
            "health": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes", "on")
    app.run(host=host, port=port, debug=debug_enabled, use_reloader=False)