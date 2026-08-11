import html
import json
import os
import time
import base64
import re
from datetime import datetime

import openai
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL_NAME, MODEL_NAME,
    QWEN_API_KEY, QWEN_BASE_URL, VL_MODEL_NAME,
    TEMP_ANALYSIS, TEMP_SEED_ROSTER, TEMP_REPORT, TEMP_CATALYST,
    TEMP_ZEITGEIST, TEMP_SANDBOX_CHAT, TEMP_DEFAULT,
)
from rag_engine import get_rag_engine, ensure_indexes
from spread_model import map_agents_to_model_params, simulate, render_spread_chart
from evaluator import evaluate


HISTORY_FILE = "history_records.json"


deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
# 向后兼容别名 — 文本调用走 DeepSeek
client = deepseek_client

# 初始化 RAG 检索引擎（首次启动自动构建 ChromaDB 索引）
ensure_indexes()


def stream_markdown_effect(container, text, prefix="", suffix="", speed=0.03):
    """模拟打字机流式输出的视觉效果"""
    placeholder = container.empty()
    displayed_text = ""
    for char in text or "":
        displayed_text += char
        placeholder.markdown(f"{prefix}{displayed_text}▌{suffix}")
        time.sleep(speed)
    placeholder.markdown(f"{prefix}{displayed_text}{suffix}")


def retrieve_similar_case(event_desc):
    """使用 RAG 向量检索替代 LLM 肉眼挑案例"""
    if not event_desc or not event_desc.strip():
        return None
    try:
        rag = get_rag_engine()
        return rag.retrieve_similar_case(event_desc)
    except Exception:
        return None


MANDATORY_CHAT_RULE = (
    "【发言风格底线】你是在社交平台（微博/小红书/B站评论区）发帖的真人，不是写公关稿的AI。"
    "严禁使用「首先其次」「作为一名」「综上所述」「高度重视」等书面/官方套话。"
    "发言应有你自己独特的声音——不同的人有不同的说话方式，不要所有人都用同一种句式。"
    "禁止所有发言以「好家伙」「绷不住了」等网络流行语开头。"
)


def _strip_todo_tags(text: str) -> str:
    """代码层过滤：移除 LLM 不小心输出的 [todo_xxx] 审查标签"""
    import re as _re
    # 匹配 [todo_数字] 或 [xxx_xxx] 格式的标签
    cleaned = _re.sub(r'\[todo_\d+\]', '', text)
    cleaned = _re.sub(r'\[[a-z]+_\d+\]', '', cleaned)
    # 清理可能残留的多余空格
    cleaned = _re.sub(r'\s{2,}', ' ', cleaned)
    return cleaned.strip()


def _history_file_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), HISTORY_FILE)


def _promote_to_case_library(case_dict):
    """将用户确认的历史测试记录提升到知识库 (case_studies.json)"""
    import shutil

    case_path = os.path.join("data", "case_studies.json")
    # 备份
    backup_path = case_path + ".bak"
    try:
        shutil.copy2(case_path, backup_path)
    except Exception:
        pass

    cases = []
    if os.path.exists(case_path):
        try:
            with open(case_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
        except Exception:
            cases = []
    if not isinstance(cases, list):
        cases = []

    # 避免重复 ID
    existing_ids = {c.get("id", "") for c in cases if isinstance(c, dict)}
    if case_dict.get("id") in existing_ids:
        case_dict["id"] = case_dict["id"] + "_" + datetime.now().strftime("%H%M%S")

    cases.append(case_dict)
    with open(case_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=4)
    print(f"[History] Promoted to case library: {case_dict.get('title', '')[:50]}")

    # 重建 ChromaDB 索引（使新案例立即可检索）
    try:
        from rag_engine import get_rag_engine
        engine = get_rag_engine()
        engine.build_index("case_studies", case_path)
    except Exception as e:
        print(f"[History] Re-index failed (non-critical): {e}")


def save_test_record(event_desc, visual_risks, sandbox_log, report_json):
    path = _history_file_path()
    clean_report = {
        k: v for k, v in report_json.items() if not str(k).startswith("_")
    }
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_desc": event_desc,
        "visual_risks": visual_risks or "",
        "sandbox_log": sandbox_log,
        "report_json": clean_report,
    }
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
        except Exception:
            records = []
    records.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def safe_json_loads(raw_text, default_value):
    text = (raw_text or "").strip()
    if not text:
        return default_value
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return default_value


def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def chat_llm(
    system_prompt,
    user_prompt,
    max_tokens,
    skip_mandatory_chat_rule=False,
    collapse_newlines=True,
):
    if skip_mandatory_chat_rule:
        final_system_prompt = system_prompt
    else:
        final_system_prompt = f"{system_prompt}\n\n{MANDATORY_CHAT_RULE}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMP_SANDBOX_CHAT,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        out = _strip_todo_tags(content or "")
        if collapse_newlines:
            out = out.replace("\n", " ")
        return out
    except Exception as e:
        return f"系统波动中，先别急！({e})"


ANALYZE_VISUAL_RISK_PROMPT = (
    "你是一位极其敏锐且言辞犀利的资深公关风控专家。请全盘审视这份视觉物料（若是视频，请重点提取核心台词/字幕、语气和人物表情）。请直接输出合法的JSON格式分析：\n"
    "{\n"
    '  "auto_event_desc": "客观描述画面内容，必须准确提取并写出视频中的核心台词或图片上的文字",\n'
    '  "auto_sentiment": "预测网民的真实情绪（如：感觉被当成狗/韭菜侮辱了、感受到极度傲慢、被激怒等）",\n'
    '  "visual_risks": "请从以下维度找出最致命的1个公关雷点并一针见血地指出：\\n1. 【最高红线：公然侮辱与品牌傲慢】：台词或文案是否直接辱骂、贬低消费者（如把消费者比作‘狗’、‘韭菜’，嘲笑消费者穷、买不起，教训消费者等）？一旦发现，必须极其严厉地定性为‘公然侮辱消费者人格与极度傲慢’，严禁使用‘拟人化’或‘情感操控’等温和词汇洗白！\\n2. 【动态行为与微表情】：是否有翻白眼、冷笑等不耐烦的肢体语言？\\n3. 【语义擦边与恶俗】：是否有低俗营销嫌疑？\\n4. 【物理造假】：是否存在极其粗劣的抠图作秀？\\n请用极其犀利的公关视角，直接输出最核心的雷点。"\n'
    "}"
)


def analyze_visual_risk(uploaded_file):
    if uploaded_file is None:
        return ""

    try:
        base64_data = encode_image(uploaded_file)
    except Exception as e:
        return f"视觉分析失败：{e}"
    if not base64_data:
        return ""

    mime_type = uploaded_file.type or "image/jpeg"
    # 使用 qwen3.5-plus OpenAI 兼容方式构建多模态消息
    if mime_type.startswith("video"):
        media_content = {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{base64_data}"},
            "fps": 2,
        }
    else:
        media_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
        }
    try:
        resp = qwen_client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位极其敏锐且言辞犀利的资深公关风控专家，擅长从视觉物料中发现致命的公关雷点。",
                },
                {
                    "role": "user",
                    "content": [
                        media_content,
                        {
                            "type": "text",
                            "text": ANALYZE_VISUAL_RISK_PROMPT,
                        },
                    ],
                },
            ],
            temperature=TEMP_ANALYSIS,
            max_tokens=4096,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 10240,
            },
        )
        raw_content = (resp.choices[0].message.content if resp.choices else "") or ""
        raw_content = raw_content.strip()
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        clean_json_str = match.group(0) if match else raw_content
        try:
            parsed = json.loads(clean_json_str)
            if isinstance(parsed, dict):
                return str(parsed.get("visual_risks", "")).strip()
        except Exception:
            pass
        return raw_content.replace("\n", " ")
    except Exception as e:
        return f"视觉分析失败：{e}"


def auto_generate_context_from_image(uploaded_file):
    if uploaded_file is None:
        st.error("视觉 API 调用失败: 未检测到上传图片")
        st.stop()

    try:
        base64_image = encode_image(uploaded_file)
    except Exception as e:
        st.error(f"视觉 API 调用失败: {str(e)}")
        st.stop()

    mime_type = uploaded_file.type or "image/jpeg"
    # 使用 qwen3.5-plus OpenAI 兼容方式构建多模态消息
    if mime_type.startswith("video"):
        media_content = {
            "type": "video_url",
            "video_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            "fps": 2,
        }
    else:
        media_content = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    auto_context_prompt = (
        "你是一位极其敏锐的资深公关风控专家。请全盘审视这份视觉物料（图片或视频片段）。如果是视频，请重点捕捉人物的微表情、语气、肢体语言以及连贯动作。请直接输出合法的JSON格式分析：\n"
        "{\n"
        '  "auto_event_desc": "客观描述物料的核心画面，若是视频请描述事件的动态过程和核心台词/字幕",\n'
        '  "auto_sentiment": "预测网民可能产生的负面情绪（如：觉得高管傲慢、觉得冷血、觉得作秀等）",\n'
        '  "visual_risks": "请从以下维度找出最致命的1个公关雷点：\\n1. 【动态行为与微表情】：视频中是否有翻白眼、指点、傲慢冷笑等不耐烦的肢体语言？\\n2. 【语义歧义与擦边】：文字或台词是否有低俗营销嫌疑？\\n3. 【物理造假】：是否存在极其粗劣的抠图或合成特效？\\n请用极其犀利的公关视角，一针见血地指出雷点。"\n'
        "}"
    )

    try:
        resp = qwen_client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "你是一位极其敏锐的资深公关风控专家，擅长从视觉物料中提取事件背景、预判网民情绪、发现公关雷点。",
                },
                {
                    "role": "user",
                    "content": [
                        media_content,
                        {
                            "type": "text",
                            "text": auto_context_prompt,
                        },
                    ],
                },
            ],
            temperature=TEMP_ANALYSIS,
            max_tokens=4096,
            extra_body={
                "enable_thinking": True,
                "thinking_budget": 10240,
            },
        )
        raw_content = (resp.choices[0].message.content if resp.choices else "") or ""
        raw_content = raw_content.strip()
    except Exception as e:
        st.error(f"视觉 API 调用失败: {str(e)}")
        st.stop()

    match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if match:
        clean_json_str = match.group(0)
    else:
        clean_json_str = raw_content

    try:
        parsed = json.loads(clean_json_str)
        if not isinstance(parsed, dict):
            raise ValueError("视觉模型返回非 JSON 对象")
        return {
            "auto_event_desc": str(parsed.get("auto_event_desc", "")).strip(),
            "auto_sentiment": str(parsed.get("auto_sentiment", "")).strip(),
            "visual_risks": str(parsed.get("visual_risks", "")).strip(),
        }
    except Exception as e:
        st.error(f"视觉 API 调用失败: JSON 解析错误 - {str(e)}\n原文: {raw_content}")
        st.stop()


PLAZA_SQUARE_RULE = (
    "【广场发帖原则】：你是在自己的社交账号上公开发帖，或者对着话题带#标签#输出情绪。"
    "严禁点对点回复特定的人（严禁使用「某某说得对」「你这是狡辩」等指向具体账号的吵架句式）！"
    "你可以引用官方通报的词眼进行嘲讽或对比，但必须是独立的观点表达，就像微博广场上的动态一样。"
    "请基于上方「实时广场热评」里的公共舆论走向发声，不要捏造不存在的热评内容。"
)
OFFICIAL_OUTPUT_FORMAT_HARD = (
    "【输出格式硬控】：你现在不是在聊天！你是在社交平台发布官方蓝底白字声明。"
    "绝对不允许出现「咱们、你、大家别急」等对话词汇。"
    "必须以【关于XXX事件的情况说明/通报】为标题开头（将XXX替换为事件核心关键词），"
    "正文使用绝对官方、冷漠或严谨的公关辞令书写（如：针对网传视频，我单位高度重视……）。"
    "字数不少于50字。"
)


def _llm_json_array(system_prompt, user_prompt, max_tokens=600, temperature=TEMP_ZEITGEIST):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content if resp.choices else "") or ""
    except Exception:
        return ""


def _is_gov_context(text):
    """检测是否为政务语境。注意：'官方''通报'太泛（每个品牌都有官方账号），已移除。"""
    keys = (
        "政务",
        "政府",
        "官员",
        "调查组",
        "纪委",
        "基层",
        "干部",
        "市委",
        "省委",
        "报案",
        "公文",
        "舆情处置",
        "宣传部",
        "党委",
        "信访",
        "行政执法",
        "城管",
        "民警",
        "派出所",
    )
    t = text or ""
    return any(k in t for k in keys)


def _is_extreme_gov_fraud(text):
    """仅当政务语境且出现明显造假/作秀信号时，允许种子中完全不设 supportive。"""
    t = text or ""
    if not _is_gov_context(t):
        return False
    fraud_keys = (
        "PS",
        "ps",
        "抠图",
        "一眼假",
        "造假",
        "合成照",
        "合成",
        "作秀",
        "摆拍",
        "糊弄",
        "政务造假",
        "通报造假",
        "假通报",
        "修图造假",
    )
    return any(k in t for k in fraud_keys)


def _is_entertainment_context(text):
    """检测是否为文娱/IP/内容创作类事件（动漫、游戏、影视、小说等）"""
    t = text or ""
    keys = (
        "动漫", "动画", "游戏", "影视", "电影", "电视剧", "综艺",
        "角色", "英雄", "人物形象", "改编", "原著", "设定",
        "粉丝", "玩家", "观众", "追番", "追剧",
        "制作组", "编剧", "导演", "CV", "配音",
        "剧情", "情节", "剧本", "台词",
        "IP", "同人", "饭圈", "塌房", "OOC",
        "漫画", "小说", "广播剧",
        "崩坏", "毁", "魔改", "诋毁", "丑化",
    )
    return any(k in t for k in keys)


def _default_supportive_fan_agent(event_desc):
    if _is_gov_context(event_desc):
        return {
            "name": "本地理性支持者",
            "persona": "仍愿意相信官方会把事情说清楚、呼吁先勿传谣的本地网民",
            "weight": 455,
            "stance": "supportive",
            "role_type": "bystander",
        }
    if _is_entertainment_context(event_desc):
        return {
            "name": "理性老观众/老玩家",
            "persona": "长期追更/游玩、对IP有深厚感情，愿意给制作组改正机会的核心受众",
            "weight": 480,
            "stance": "supportive",
            "role_type": "bystander",
        }
    return {
        "name": "十年老粉用户",
        "persona": "长期复购、对品牌有感情的消费者，遇事本能想先替品牌说两句",
        "weight": 520,
        "stance": "supportive",
        "role_type": "bystander",
    }


def generate_dynamic_catalyst_agents(event_desc, network_mood):
    system_prompt = (
        "你是舆情沙盘编剧，只输出合法 JSON。"
        "设计恰好 2 个角色：\n"
        "第1个：分析型评论员——能跳出情绪、从行业/制度层面冷静分析事件，提供第三方客观视角。"
        "类似：行业观察者、资深媒体人、公关/管理专家。\n"
        "第2个：对抗型高影响力角色——在舆论场中带节奏、煽动情绪的意见领袖。"
        "类似：考据博主、CP粉头、维权大V、毒舌吐槽博主。\n"
        "姓名与人设必须贴合当前具体事件，禁止套用无关模板。\n"
        "输出格式：严格 JSON 数组，长度恰好 2，每项仅含 name、persona。禁止 Markdown。"
    )
    user_prompt = (
        f"事件描述：{event_desc}\n"
        f"网络情绪：{network_mood}\n"
        "请输出上述 JSON 数组。第1个为分析型，第2个为对抗型。"
    )
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=400, temperature=TEMP_CATALYST)
    roster = safe_json_loads(raw, default_value=[])
    if not isinstance(roster, list):
        roster = []
    out = []
    for item in roster:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        persona = str(item.get("persona", "")).strip()
        if name and persona:
            out.append({"name": name, "persona": persona})
        if len(out) >= 2:
            break
    if len(out) < 2:
        if _is_gov_context(event_desc):
            out = [
                {"name": "资深媒体观察员", "persona": "长期跟踪公共事件舆情，善于从制度和流程角度冷静分析，不站队不煽情"},
                {"name": "较真考据网友", "persona": "逐帧抠细节、专盯时间线与证据链的技术型网民"},
            ]
        elif _is_entertainment_context(event_desc):
            out = [
                {"name": "行业观察者·阿镜", "persona": "关注国漫产业多年的评论人，能从行业惯例和制作流程角度分析是非，给出独立判断"},
                {"name": "毒舌吐槽博主", "persona": "专做动漫游戏吐槽、擅长用截图对比制造传播爆点的娱乐评论账号"},
            ]
        else:
            out = [
                {"name": "资深品牌顾问", "persona": "有多年公关咨询经验，能从危机管理和品牌策略角度冷静分析得失"},
                {"name": "高活跃维权大V", "persona": "带动消费者集体投诉与话题冲榜的意见领袖"},
            ]
    return out[:2]


def generate_seed_roster(event_desc, network_mood, pr_draft, rag_result=None):
    # Build RAG context for master agent — v2.0 分类框架优先
    rag_context_block = ""
    if rag_result:
        # v2.0: 从 rag_engine 获取分类框架上下文
        try:
            rag = get_rag_engine()
            rag_context_block = rag.format_master_agent_context(rag_result)
        except Exception:
            pass

        # 兼容旧 RAG 证据（红线/黑词简短版）
        if not rag_context_block:
            red_lines = rag_result.get("red_lines", [])
            similar = rag_result.get("similar_cases", [])
            black_words = rag_result.get("black_words", [])
            if red_lines:
                rl_hit = red_lines[0]
                rag_context_block += (
                    "Redline hit: " + str(rl_hit.get("source","")) + " [" + str(rl_hit.get("category",""))
                    + "] severity=" + str(rl_hit.get("severity","?")) + "/5. "
                )
            if black_words:
                bw_hits = [str(bw.get("word","")) for bw in black_words[:3]]
                rag_context_block += "High-risk phrases: " + ", ".join(bw_hits) + "."

    system_prompt = (
        "你是舆情模拟引擎，只输出合法 JSON，禁止 Markdown。\n"
        "你的任务：根据事件描述和知识库分析，设计恰好 3 个模拟社交平台用户。\n\n"
        "【核心要求】姓名(name)和人设(persona)必须贴合这个具体事件，禁止套用通用模板。\n"
        "角色名必须带事件特征——比如动漫事件的角色名应该是'追番老粉''角色考据党'而非'消费者代表'。\n\n"
        "【3个角色必须包含】\n"
        "1. 1名 official（官方回应者）——代表涉事方对外发声\n"
        "2. 1名 hostile（批评者/愤怒受众）——从事件受害方角度攻击\n"
        "3. 1名 supportive（理性支持者/愿意给机会的人）——不是无脑维护而是抱有善意但也要看到行动\n\n"
        "每个角色字段：name(中文，≤6字), persona(一句话人设), "
        "weight(100-1000整数), stance(supportive/neutral/hostile), "
        "role_type(official/influencer/bystander)。\n\n"
        "返回格式（严格 JSON）：\n"
        '{"roster":['
        '{"name":"...","persona":"...","weight":600,"stance":"neutral","role_type":"official"},'
        '{"name":"...","persona":"...","weight":600,"stance":"hostile","role_type":"bystander"},'
        '{"name":"...","persona":"...","weight":500,"stance":"supportive","role_type":"bystander"}'
        ']}'
    )
    user_prompt = (
        f"事件描述：{event_desc}\n"
        f"网络情绪：{network_mood}\n"
        f"公关草稿：{pr_draft}\n"
        f"系统知识库分析：{rag_context_block or '无特殊命中'}\n"
        "请输出上述 JSON 数组。角色名和人设必须贴合这个具体事件。"
    )
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=500, temperature=TEMP_SEED_ROSTER)
    parsed = safe_json_loads(raw, default_value={})

    master_agent_reasoning = {}
    roster = []

    if isinstance(parsed, dict):
        # 新格式: {"roster": [...]} 或旧格式: {"master_agent_reasoning": ..., "roster": [...]}
        roster = parsed.get("roster") or []
        if not roster and "master_agent_reasoning" in parsed:
            roster = parsed.get("roster") or []
    elif isinstance(parsed, list):
        roster = parsed

    if not roster:
        # 最后一次尝试：直接让 LLM 生成纯角色数组
        retry_prompt = (
            "只输出 JSON 数组，包含 3 个角色对象。每个对象有 name, persona, weight, stance, role_type 字段。"
        )
        retry_raw = _llm_json_array(retry_prompt, user_prompt, max_tokens=400, temperature=0.3)
        retry_parsed = safe_json_loads(retry_raw, default_value=[])
        if isinstance(retry_parsed, list):
            roster = retry_parsed
        elif isinstance(retry_parsed, dict):
            roster = retry_parsed.get("roster", [])

    cleaned = []
    for item in roster:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip() or "匿名角色"
        persona = str(item.get("persona", "")).strip() or "围观群众"
        stance = str(item.get("stance", "neutral")).strip().lower()
        if stance not in {"supportive", "neutral", "hostile"}:
            stance = "neutral"
        role_type = str(item.get("role_type", "bystander")).strip().lower()
        if role_type not in {"official", "influencer", "bystander"}:
            role_type = "bystander"
        try:
            weight = int(item.get("weight", 300))
        except Exception:
            weight = 300
        weight = max(100, min(1000, weight))
        cleaned.append(
            {
                "name": name,
                "persona": persona,
                "weight": weight,
                "stance": stance,
                "role_type": role_type,
            }
        )

    # ── 代码层强制 stance/role 分布 ──
    # LLM 经常忽略 prompt 中的 stance 分布要求，这里兜底修正
    has_official = any(a.get("role_type") == "official" for a in cleaned)
    has_supportive = any(a.get("stance") == "supportive" for a in cleaned)
    has_hostile = any(a.get("stance") == "hostile" for a in cleaned)

    if not _is_extreme_gov_fraud(event_desc) and cleaned:
        # 1. 确保有 supportive
        if not has_supportive:
            fan = _default_supportive_fan_agent(event_desc)
            if len(cleaned) >= 3:
                cleaned[-1] = fan
            else:
                cleaned.append(fan)
            has_supportive = True

        # 2. 确保有 official（替换第一个非 official 角色）
        if not has_official and len(cleaned) >= 2:
            # 找到第一个 bystander 替换为 official
            for i, a in enumerate(cleaned):
                if a.get("role_type") == "bystander":
                    cleaned[i] = {
                        "name": "制作组/官方账号" if _is_entertainment_context(event_desc)
                                else "品牌公关",
                        "persona": "负责对外发声的内容方代表" if _is_entertainment_context(event_desc)
                                   else "试图稳住舆论、统一口径的品牌方发言人",
                        "weight": 640,
                        "stance": "neutral",
                        "role_type": "official",
                    }
                    has_official = True
                    break
            # 如果全是 influencer，改第一个为 official
            if not has_official:
                cleaned[0] = {
                    "name": "制作组/官方账号" if _is_entertainment_context(event_desc)
                            else "品牌公关",
                    "persona": "负责对外发声的内容方代表" if _is_entertainment_context(event_desc)
                               else "试图稳住舆论、统一口径的品牌方发言人",
                    "weight": 640,
                    "stance": "neutral",
                    "role_type": "official",
                }

    if len(cleaned) < 2:
        if _is_gov_context(event_desc):
            cleaned = [
                {
                    "name": "当地宣传部门工作人员",
                    "persona": "负责对外口径与舆情回应的政务发布角色",
                    "weight": 620,
                    "stance": "neutral",
                    "role_type": "official",
                },
                {
                    "name": "涉事单位回应人",
                    "persona": "事件直接责任方指定的对外说明人员",
                    "weight": 680,
                    "stance": "neutral",
                    "role_type": "official",
                },
                {
                    "name": "本地理性支持者",
                    "persona": "仍愿意相信调查会把事情说清楚、呼吁先勿传谣的本地网民",
                    "weight": 450,
                    "stance": "supportive",
                    "role_type": "bystander",
                },
            ]
        elif _is_entertainment_context(event_desc):
            cleaned = [
                {
                    "name": "制作组/官方账号",
                    "persona": "负责对外发声的内容方代表，需要回应粉丝和舆论关切",
                    "weight": 640,
                    "stance": "neutral",
                    "role_type": "official",
                },
                {
                    "name": "失望的核心粉丝",
                    "persona": "长期关注IP、对角色和剧情有很深感情的观众/玩家，因内容质量或价值观问题感到被背叛",
                    "weight": 620,
                    "stance": "hostile",
                    "role_type": "bystander",
                },
                {
                    "name": "理性老观众/老玩家",
                    "persona": "愿意给制作组改正机会、但要求看到具体行动而非话术的资深受众",
                    "weight": 480,
                    "stance": "supportive",
                    "role_type": "bystander",
                },
            ]
        else:
            cleaned = [
                {
                    "name": "品牌公关",
                    "persona": "试图稳住舆论、统一口径的品牌方发言人",
                    "weight": 640,
                    "stance": "neutral",
                    "role_type": "official",
                },
                {
                    "name": "消费者代表",
                    "persona": "因产品或服务体验受损而发声的用户",
                    "weight": 610,
                    "stance": "hostile",
                    "role_type": "bystander",
                },
                {
                    "name": "十年老粉用户",
                    "persona": "长期复购、对品牌有感情，遇事本能想先替品牌说两句的用户",
                    "weight": 500,
                    "stance": "supportive",
                    "role_type": "bystander",
                },
            ]

    mr = master_agent_reasoning if isinstance(master_agent_reasoning, dict) else {}
    return {
        "master_agent_reasoning": {
            "trigger_anchor": str(mr.get("trigger_anchor", "") or "").strip(),
            "rag_evidence": str(mr.get("rag_evidence", "") or "").strip(),
            "strategy_argument": str(mr.get("strategy_argument", "") or "").strip(),
            "evolution_prediction": str(mr.get("evolution_prediction", "") or "").strip(),
        },
        "roster": cleaned[:3],
    }


def render_master_agent_dashboard(master_agent_reasoning):
    mr = master_agent_reasoning or {}
    crisis_type = str(mr.get("crisis_type", "") or "").strip() or "（未识别）"
    trigger_anchor = str(mr.get("trigger_anchor", "") or "").strip() or "（未提供锚点）"
    rag_evidence = str(mr.get("rag_evidence", "") or "").strip() or "（未提供论据）"
    strategy_argument = (
        str(mr.get("strategy_argument", "") or "").strip() or "（未提供论点）"
    )
    evolution_prediction = str(mr.get("evolution_prediction", "") or "").strip() or "（未预判）"

    st.markdown("### 🧠 Master Agent 逻辑自证中枢")
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"🏷️ **危机类型判定**\n\n{crisis_type}")
        with col2:
            st.info(f"📍 **核心语义锚点**\n\n{trigger_anchor}")
        with col3:
            st.warning(f"📚 **分类框架匹配**\n\n{rag_evidence}")
        with col4:
            st.success(f"⚙️ **阵营推演策略**\n\n{strategy_argument}")

        st.markdown(
            f"""
```log
[Taxonomy] 危机分类框架命中 -> {crisis_type}
[Trigger]  多模态语义锚点定位 -> {trigger_anchor}
[RAG]      知识库模式匹配完成 -> {rag_evidence}
[Strategy] 角色阵营装载执行   -> {strategy_argument}
[Predict]  舆情演化路径预判   -> {evolution_prediction}
[Status]   逻辑自证闭环完成。沙盘推演引擎启动...
```
"""
        )


def analyze_macro_zeitgeist(event_desc, visual_risks, hot_topics):
    system_prompt = (
        "你是一个精通社会心理学的宏观舆情分析师。请结合用户输入的‘事件背景’和‘当前互联网敏感热点’，"
        "分析该公关行为是否会意外触碰网民近期的‘创伤记忆’或‘敏感神经’（例如：在全网声讨学术造假的节骨眼上营销天才学霸）。"
        "请直接输出 JSON: "
        '{"risk_level": "高/中/低", "collateral_damage_warning": "一句话概括是否撞枪口及原因"}'
    )
    ht = hot_topics.strip() if hot_topics else "（用户未填写，请结合近期常见舆情节律自行推断）"
    user_prompt = (
        f"事件背景：{event_desc}\n"
        f"视觉雷点摘要：{visual_risks or '无'}\n"
        f"当前互联网敏感热点（用户填写）：{ht}\n"
        "请只输出上述 JSON 对象，不要 Markdown。"
    )
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=400, temperature=TEMP_REPORT)
    cleaned = (raw or "").replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(0)
    fallback = {"risk_level": "低", "collateral_damage_warning": ""}
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return fallback
        rl = str(parsed.get("risk_level", "低")).strip()
        if rl not in {"高", "中", "低"}:
            rl = "低"
        warn = str(parsed.get("collateral_damage_warning", "") or "").strip()
        return {"risk_level": rl, "collateral_damage_warning": warn}
    except Exception:
        return fallback


def build_plaza_context(event_desc, logs_rows):
    lines = [
        f"当前热搜话题：{event_desc}",
    ]
    # 把最新官方通报置顶，确保 Agent 看到
    latest_official = None
    other_rows = []
    for row in reversed(logs_rows):
        if row.get("role_type") == "official" and latest_official is None:
            latest_official = row
        else:
            other_rows.append(row)
    other_rows.reverse()

    if latest_official:
        lines.append("--- 📢 最新官方通报（请仔细阅读后回应） ---")
        lines.append(f"[官方 @{latest_official.get('name','')}]：{latest_official.get('speech','')}")
        lines.append("--- 实时广场热评 ---")

    if not logs_rows:
        lines.append("（广场暂无高赞摘选，本话题刚开始发酵。）")
    else:
        for i, row in enumerate(other_rows[-7:], 1):  # 最近 7 条非官方发言
            nick = row.get("name", "网友")
            speech = row.get("speech", "")
            lines.append(f"[热评{i} - @{nick}]：{speech}")
    return "\n".join(lines)


def render_official_announcement(agent, text, round_idx, phase_label):
    st.markdown(
        f"**📢 官方蓝底白字通报** ｜ `{html.escape(agent.get('name', ''))}` ｜ "
        f"Round {round_idx} ｜ {phase_label}"
    )
    safe = html.escape(text or "")
    st.markdown(
        f"""
<div style="background:#1e3a8a;color:#f8fafc;padding:16px 18px;border-radius:8px;
border-left:8px solid #60a5fa;margin:8px 0 20px 0;font-size:15px;line-height:1.65;
box-shadow:0 2px 8px rgba(30,58,138,0.35);">
<div style="white-space:pre-wrap;">{safe}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_agent_bubble(agent, text, round_idx):
    rt = agent.get("role_type", "bystander")
    header = (
        f"**🧠 {agent['name']}** ｜ 人设：`{agent['persona']}` ｜ 权重：`{agent['weight']}` "
        f"｜ 立场：`{agent['stance']}` ｜ 角色：`{rt}` ｜ 回合：`Round {round_idx}` ｜ 广场动态"
    )
    safe = html.escape(text or "")
    st.markdown(header)
    st.markdown(
        f"""
<div style="background:#111827;padding:10px 12px;border-radius:10px;margin:6px 0 14px 0;">
  <span style="color:#F9FAFB;font-size:16px;">{safe}</span>
</div>
""",
        unsafe_allow_html=True,
    )



def _build_rag_context_for_role(rag_result, agent_role):
    """
    v2.0: 根据 Agent 角色构建 RAG 上下文。
    优先使用分类框架 + 审查清单，降级为传统红线/黑词/恶评。
    - official: 审查清单(阶段1-4) + 分类框架 + 红线
    - influencer: 定性+找茬清单(阶段1-3) + 分类框架 + 恶评风格
    - bystander: 感受清单(阶段2-3) + 分类框架 + 恶评风格
    """
    if not rag_result:
        return ""
    try:
        rag = get_rag_engine()
        ctx = rag.format_context_for_agent(rag_result)
        if ctx.strip():
            return ctx
    except Exception:
        pass

    # 降级：使用传统格式化
    parts = []
    red_lines = rag_result.get("red_lines", [])
    if red_lines and agent_role == "official":
        rl_texts = []
        for rl in red_lines[:3]:
            rl_texts.append(
                "• " + str(rl.get("source","")) + " [" + str(rl.get("category","")) + "] "
                + "severity=" + str(rl.get("severity","?")) + "/5"
                + " - penalty: " + str(rl.get("penalty",""))
            )
        parts.append("【法律红线警告】你的通报措辞必须规避：\n" + "\n".join(rl_texts))

    hate_comments = rag_result.get("hate_comments", [])
    if hate_comments and agent_role != "official":
        hc_texts = []
        for hc in hate_comments[:5]:
            hc_texts.append(
                "• 「" + str(hc.get("content","")) + "」(likes:" + str(hc.get("likes_count","?"))
                + " emotion:" + str(hc.get("emotion_type","")) + ")"
            )
        parts.append("【网民恶评风格参考】吸收修辞风格与情绪切入角度，但不要原样复制：\n" + "\n".join(hc_texts))

    black_words = rag_result.get("black_words", [])
    if black_words:
        bw_texts = []
        for bw in black_words[:3]:
            bw_texts.append(
                "• 「" + str(bw.get("word","")) + "」 - "
                + str(bw.get("why_it_exploded",""))[:120]
            )
        parts.append("【高危措辞预警】请规避以下词汇/句式：\n" + "\n".join(bw_texts))
    return "\n\n".join(parts)

ROUND4_FINAL_BEHAVIOR_LAW = (
    "【终局行为法则（反杠精警告）】：\n"
    "请你深呼吸，**逐字阅读**刚刚官方发布的蓝底白字通报！绝对不要只盯着‘操作失误’或‘新人’等字眼找茬，"
    "你必须重点审视他们的【最终解决方案】！\n"
    "🟢 判定为【神级/正向公关】的条件：如果官方承诺了‘履行所有异常订单’、‘按错价发货’、‘全额赔偿’或‘承担巨大损失’。"
    "这说明品牌方格局极大，极其真诚！\n"
    "  -> 若你被设定为 hostile (敌对大V/路人)：你必须表现出‘被打动’、‘被折服’或‘路转粉’。"
    "你的发言必须转为赞赏（如：‘真发货啊？这格局我服了’、‘这公关可以，我闭嘴’、‘活该你赚钱’），"
    "**严禁再骂官方甩锅或玩不起！**\n"
    "  -> 若你被设定为 supportive (粉丝)：你必须感到极其骄傲，扬眉吐气地大声称赞品牌。\n"
    "🔴 判定为【灾难/负向公关】的条件：如果官方的解决方案是‘强制取消订单’、‘仅道歉不赔偿’、"
    "‘闭口不谈发货’或态度强硬。\n"
    "  -> 所有人（包括粉丝）都必须表现出极度的愤怒、失望，继续猛烈嘲讽。\n\n"
    "请根据上述法则，给出你最符合人类真实心理的反应！"
)

ROUND4_ANTI_COLLISION_TAIL = (
    "【防撞车警告】：严禁重复前面人说过的梗和句式！前面人如果夸了格局，你就夸契约精神或表示要野性消费，"
    "每个人必须有完全不同的视角！"
)


def render_cyber_graph(reasoning_data):
    """渲染带有动态点亮效果的 Master Agent 赛博节点图"""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ background-color: transparent; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; margin: 0; }}
        .container {{ position: relative; width: 100%; height: 450px; display: flex; justify-content: center; align-items: center; background: #0e1117; border-radius: 12px; box-shadow: inset 0 0 40px rgba(0,0,0,0.5); border: 1px solid #1e2532; }}
        .node {{ position: absolute; padding: 15px; border-radius: 10px; background: rgba(22, 30, 45, 0.9); border: 1px solid #3b82f6; opacity: 0; transition: all 0.8s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 0 15px rgba(59, 130, 246, 0.2); max-width: 220px; font-size: 13px; line-height: 1.5; backdrop-filter: blur(4px); }}
        .node.active {{ opacity: 1; transform: scale(1); }}
        .node h4 {{ margin: 0 0 8px 0; color: #60a5fa; font-size: 14px; display: flex; align-items: center; gap: 6px; }}
        .center-node {{ z-index: 10; padding: 20px 25px; border-radius: 50px; background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); border: 2px solid #3b82f6; box-shadow: 0 0 30px rgba(59, 130, 246, 0.5); font-weight: bold; font-size: 16px; text-align: center; color: white; }}
        
        /* 节点分布定位 */
        .step1 {{ top: 8%; left: 50%; transform: translateX(-50%) scale(0.5); }}
        .step2 {{ top: 40%; right: 5%; transform: translateY(-50%) scale(0.5); }}
        .step3 {{ bottom: 8%; left: 50%; transform: translateX(-50%) scale(0.5); }}
        .step4 {{ top: 40%; left: 5%; transform: translateY(-50%) scale(0.5); }}
        
        /* 动态连线 SVG */
        svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }}
        line {{ stroke: #3b82f6; stroke-width: 2; stroke-dasharray: 6, 6; opacity: 0; transition: opacity 0.8s; }}
        .line-active {{ opacity: 0.4; animation: dash 20s linear infinite; }}
        @keyframes dash {{ to {{ stroke-dashoffset: -1000; }} }}
    </style>
    </head>
    <body>
        <div class="container">
            <svg>
                <line id="line1" x1="50%" y1="50%" x2="50%" y2="25%" />
                <line id="line2" x1="50%" y1="50%" x2="75%" y2="40%" />
                <line id="line3" x1="50%" y1="50%" x2="50%" y2="75%" />
                <line id="line4" x1="50%" y1="50%" x2="25%" y2="40%" />
            </svg>

            <div class="center-node">🧠<br>总控智能体<br><span style="font-size:10px; color:#94a3b8;">Master Agent</span></div>
            
            <div class="node step1" id="node1">
                <h4>👁️ 1. 场景判别</h4>
                <div>{reasoning_data.get('trigger_anchor', '解析中...')}</div>
            </div>
            <div class="node step2" id="node2">
                <h4>📚 2. 论据支撑</h4>
                <div>{reasoning_data.get('rag_evidence', '关联中...')}</div>
            </div>
            <div class="node step3" id="node3">
                <h4>⚙️ 3. 论点推断</h4>
                <div>{reasoning_data.get('strategy_argument', '生成中...')}</div>
            </div>
            <div class="node step4" id="node4">
                <h4>⚠️ 4. 冲突焦点</h4>
                <div>{reasoning_data.get('evolution_prediction', '正在计算分支...')}</div>
            </div>
        </div>

        <script>
            // 依次延时点亮节点和连线
            setTimeout(() => {{ document.getElementById('node1').classList.add('active'); document.getElementById('line1').classList.add('line-active'); }}, 600);
            setTimeout(() => {{ document.getElementById('node2').classList.add('active'); document.getElementById('line2').classList.add('line-active'); }}, 1800);
            setTimeout(() => {{ document.getElementById('node3').classList.add('active'); document.getElementById('line3').classList.add('line-active'); }}, 3000);
            setTimeout(() => {{ document.getElementById('node4').classList.add('active'); document.getElementById('line4').classList.add('line-active'); }}, 4200);
        </script>
    </body>
    </html>
    """
    # 渲染高度设为 470，匹配内部容器高度
    components.html(html_code, height=470)


def run_dynamic_sandbox(
    event_desc,
    network_mood,
    pr_draft,
    agents,
    zeitgeist_result=None,
    matched_case=None,
    rag_result=None,
):
    logs = []
    primary_catalyst_name = ""
    catalyst_names = set()

    # RAG context snippets for different agent types
    rag_context = _build_rag_context_for_role(rag_result, "official") if rag_result else ""
    rag_crowd_context = _build_rag_context_for_role(rag_result, "influencer") if rag_result else ""

    for round_idx in range(1, 5):
        st.markdown(f"### Round {round_idx}")

        if round_idx == 2:
            catalyst_specs = generate_dynamic_catalyst_agents(event_desc, network_mood)
            injected = []
            for i, spec in enumerate(catalyst_specs):
                # 第1个催化剂为分析型(neutral)，第2个为对抗型(hostile)
                injected.append(
                    {
                        "name": spec["name"],
                        "persona": spec["persona"],
                        "weight": 850 if i == 0 else 700,
                        "stance": "neutral" if i == 0 else "hostile",
                        "role_type": "influencer",
                    }
                )
            agents.extend(injected)
            if injected:
                primary_catalyst_name = injected[0]["name"]
                catalyst_names = {a["name"] for a in injected}
            st.warning(
                "🔥 舆情破圈！分析评论员「{0}」与对抗节点「{1}」加入讨论".format(
                    injected[0]['name'], injected[1]['name'])
            )

        latest_catalyst_speech = ""
        if round_idx >= 3 and primary_catalyst_name:
            for row in reversed(logs):
                if row["round"] == (round_idx - 1) and row["name"] == primary_catalyst_name:
                    latest_catalyst_speech = row["speech"]
                    break
            if not latest_catalyst_speech and catalyst_names:
                for row in reversed(logs):
                    if row["round"] == (round_idx - 1) and row["name"] in catalyst_names:
                        latest_catalyst_speech = row["speech"]
                        break

        current_round_responses = []

        for agent in agents:
            rt_agent = agent.get("role_type", "bystander")
            if rt_agent == "official" and round_idx not in (1, 4):
                continue

            context_text = build_plaza_context(event_desc, logs[-8:])

            if round_idx == 3 and rt_agent != "official":
                context_text += (
                    "\n--- 广场态势 ---\n"
                    "本话题下，上一轮由头部对立账号带起的代表性高赞口径已在广场出现（见上方热评摘选）。"
                    "请继续围绕公共话题与证据链条独立发帖，带话题标签式输出情绪，严禁点名@人或回贴式吵架。"
                )
                if latest_catalyst_speech:
                    context_text += (
                        "\n（广场高热切口摘录，仅供对照舆论焦点，禁止据此点对点回怼。）\n「"
                        + latest_catalyst_speech
                        + "」"
                    )

            if round_idx == 4:
                if rt_agent == "official":
                    context_text += (
                        "\n【结局回合·官方】：发布最终蓝底白字通报，正视舆情、不得退缩或敷衍跑路。"
                        "须体现调查进展或处置安排等严肃信息，仍须严格遵守输出格式硬控。"
                    )
                elif rt_agent == "influencer":
                    context_text += "\n【最高指令】：给出你最后的总结性施压（广场发帖，不@人）。"
                else:
                    context_text += "\n【最高指令】：吃瓜疲劳，留下一句退场白（广场发帖，不@人）。"

            macro_bg = ""
            if zeitgeist_result and str(zeitgeist_result.get("risk_level", "")).strip() == "高":
                zw = (zeitgeist_result.get("collateral_damage_warning") or "").strip()
                if zw:
                    macro_bg = (
                        "【大环境背景】：目前互联网上刚发生过类似敏感事件（如近期热议："
                        f"{zw}"
                        "），这让网民对此类话题极度敏感和暴躁。"
                        "请在你的文本中自然映射或借题该氛围（官方通报须克制、可点到为止提及高度重视舆情反映）。"
                    )

            if rt_agent == "official":
                system_prompt = (
                    "【审查框架 · 官方通报撰写指南】\n"
                    "请参照上方「审查清单」逐项检查你的通报文本。重点注意：\n"
                    "1. 是否有「甩锅」措辞？（外包/临时工/第三方/个别员工）→ 绝对禁用\n"
                    "2. 是否有「傲慢」措辞？（教育消费者/嫌贵/层次）→ 绝对禁用\n"
                    "3. 是否有「模板化」措辞？（高度重视/深表歉意/举一反三不带行动）→ 需要具体化\n"
                    "4. 危机模式匹配告知：请阅读上方「危机模式匹配」部分，确认你理解了当前事件的危机类型\n"
                    "5. 按照「审查清单」第4阶段的策略建议输出改写方案\n"
                    "【严禁】在通报正文中输出 [todo_xxx] 编号标签，那会毁掉官方声明的严肃性。\n"
                    "\n"
                    + f"你执笔对外口径的官方身份：{agent['name']}（{agent['persona']}）。"
                    + "你只负责机构通报文本，不参与网民互怼，也不使用聊天语气。"
                    + f"{OFFICIAL_OUTPUT_FORMAT_HARD}"
                    + f"{macro_bg}"
                    + f"{rag_context}"
                )
                if round_idx == 1:
                    round_instruction = (
                        "Round1 初步定调：请发布首份【关于XXX事件的情况说明/通报】，阐明已关注、正核查、将及时通报等官方态度。"
                    )
                else:
                    round_instruction = (
                        "Round4 最终通报：请发布终版蓝底白字通报，可回应调查进展、处置措施与下一步安排，态度冷静严谨。"
                    )
                user_prompt = (
                    f"{context_text}\n"
                    f"网络情绪总览：{network_mood}\n"
                    f"内部参考草稿（勿口语化照抄，须改写为正式通报体）：{pr_draft}\n"
                    f"{round_instruction}"
                )
                speech = chat_llm(
                    system_prompt,
                    user_prompt,
                    max_tokens=600,
                    skip_mandatory_chat_rule=True,
                    collapse_newlines=False,
                )
                phase = "初步定调" if round_idx == 1 else "最终通报"
                render_official_announcement(agent, speech, round_idx, phase)
            else:
                # ── 角色差异化发言系统 ──
                pressure_label = {1: "初期", 2: "升温", 3: "爆发", 4: "终局"}.get(round_idx, "")
                agent_stance = agent.get("stance", "neutral")
                agent_name = agent.get("name", "")
                agent_persona = agent.get("persona", "")

                spiral_snippet = (
                    f"【舆论态势·Round{round_idx}】当前处于话题{pressure_label}阶段。"
                )

                # ── 立场专属发言约束（风格 + 内容双重差异化） ──
                if agent_stance == "supportive":
                    spiral_snippet += (
                        "\n【你的立场·supportive】你是愿意给机会的理性支持者。"
                        "你的声音在广场上是少数派，表达压力会随轮次增加——"
                        "但你始终从「建设性」角度发言。\n"
                        "【强制要求】你的发言必须同时包含两部分：\n"
                        "  (1) 先指出官方回应中做得对/有进步的一个具体点（找不到也要找——至少他们回应了/下架了/道歉了）\n"
                        "  (2) 再提出一个具体的改进期望（不是笼统的'要改进'，是你希望看到什么）\n"
                        "【语气】温和但有立场，像朋友间的认真建议，不使用反问句、不使用讽刺。\n"
                        "【句式禁令】禁止说'但'后面跟攻击性内容把自己变成 hostile。禁止说'笑死''好家伙''绷不住了'。"
                    )
                elif agent_stance == "hostile":
                    # 根据 persona 分配不同的切入角度
                    persona_lower = agent_persona.lower()
                    if "历史" in persona_lower or "考据" in persona_lower or "史料" in persona_lower:
                        angle = "历史考据角度——用史料和事实反驳官方回应中不符合历史的设定"
                    elif "cp" in persona_lower or "小乔" in persona_lower or "瑜乔" in persona_lower:
                        angle = "CP/角色情感角度——从角色关系和粉丝情感投入的角度质疑"
                    elif "维权" in persona_lower or "粉头" in persona_lower or "号召" in persona_lower:
                        angle = "消费者权益角度——从玩家/观众作为付费用户的权益角度施压"
                    elif "吐槽" in persona_lower or "毒舌" in persona_lower or "鉴抄" in persona_lower:
                        angle = "行业对比角度——用同行业其他案例对比揭短"
                    else:
                        angle = "责任追问角度——追问官方回应中未回答的核心问题"
                    spiral_snippet += (
                        f"\n【你的立场·hostile】你是批评者。但你绝不是无脑喷子——你的火力有明确方向。\n"
                        f"【你的专属角度】{angle}\n"
                        f"【约束】严守你的角度，不要越界去说其他角度的话（其他人在说）。"
                        f"批评要具体到官方回应的哪句话、哪个词有问题。不使用纯情绪词汇堆砌。"
                    )
                elif agent_stance == "neutral":
                    spiral_snippet += (
                        "\n【你的立场·neutral】你是冷静的第三方分析者——不站队任何一方。"
                        "你的视角是「这件事在行业里意味着什么」「为什么会发生」「怎么避免」。"
                        "你不是来骂人的，也不是来维护的，你是来解释和预判的。"
                        "你可以承认官方回应中做得对的部分，也可以指出粉丝诉求中不切实际的地方。"
                        "【语气】专业但不冷漠，像资深媒体人或行业分析师在写评论。不使用反问句和讽刺。"
                    )

                # ── 通用约束 ──
                spiral_snippet += (
                    "\n【事实约束】只评论当前事件中真实存在的内容。禁止编造不存在的第三方/事件。"
                )
                # 后期轮次：优先回应最新官方通报
                if round_idx >= 3:
                    spiral_snippet += (
                        f"\n【回应最新通报】R{round_idx-1}发布了最新官方通报（见上方蓝底白字）。"
                        "你的发言必须针对这则最新通报中的具体措辞。"
                    )

                round4_law = ROUND4_FINAL_BEHAVIOR_LAW if round_idx == 4 else ""
                system_prompt = (
                    f"你是 {agent['name']}（{agent['persona']}）。"
                    + f"立场：{agent_stance}。"
                    + "你在社交平台公开发帖。\n"
                    + "【核心原则】你是这个具体的人，有你自己独特的说话方式。"
                    + "不要和其他人用相同的句式、相同的开头、相同的梗。"
                    + "如果有人的角度已经和你接近，你就换个切入点——但不是换个词说同样的话。"
                    + "禁止输出 [todo_xxx] 标签。说人话。"
                    + f"{macro_bg}"
                    + f"{spiral_snippet}"
                    + f"{round4_law}"
                    + f"{rag_crowd_context}"
                )
                # ── 轮次指令（含字数限制） ──
                if round_idx == 1:
                    round_instruction = (
                        "Round1：试探性带话题发帖。控制在80-100字，像真实社交平台发言，不要写成小作文。"
                    )
                elif round_idx == 2:
                    round_instruction = (
                        "Round2：话题升温，强化观点。控制在80-100字，一句核心观点+一个具体论据即可。"
                    )
                elif round_idx == 3:
                    round_instruction = (
                        "Round3：焦灼对撞期。控制在100字以内。"
                        "【跨视角引用】如果广场上已有人从其他角度发了言，"
                        "你可以先一句话点出你和他的异同（如'考据派说得对，但从CP粉角度看这更气人——'），"
                        "然后展开你的补充。不是重复他，是在他的基础上延伸你自己的角度。"
                    )
                elif round_idx == 4:
                    round_instruction = (
                        "Round4：结局回合。可以稍长但不超过150字。"
                        "总结你从R1到现在的核心立场，针对最终通报给出你的最终判断。"
                    )

                anti_echo = ""
                if current_round_responses:
                    joined = "\n".join(current_round_responses)
                    anti_echo = (
                        "\n【防雷同警告】广场上本轮已出现的其他声音：\n"
                        f"{joined}\n"
                        "你绝对不能重复上述声音的句式和核心词。找别人没碰过的切入点。"
                    )

                user_prompt = (
                    f"{context_text}\n"
                    f"网络情绪：{network_mood}\n"
                    f"涉事方公开表态参考：{pr_draft}\n"
                    f"{round_instruction}\n"
                    f"{PLAZA_SQUARE_RULE}"
                    f"{anti_echo}"
                )
                if round_idx == 4:
                    user_prompt += "\n" + ROUND4_ANTI_COLLISION_TAIL
                # 轮次相关的 token 预算
                r_tokens = {1: 120, 2: 120, 3: 120, 4: 180}.get(round_idx, 120)
                speech = chat_llm(system_prompt, user_prompt, max_tokens=r_tokens)
                render_agent_bubble(agent, speech, round_idx)

            logs.append(
                {
                    "round": round_idx,
                    "name": agent["name"],
                    "persona": agent["persona"],
                    "weight": agent["weight"],
                    "stance": agent["stance"],
                    "role_type": agent.get("role_type", "bystander"),
                    "speech": speech,
                    "ts": int(time.time()),
                }
            )
            current_round_responses.append(f"「{speech}」")


    return logs


def generate_report(event_desc, network_mood, pr_draft, visual_risk_desc, logs,
                    rag_result=None, spread_result=None):
    """混合评估：确定性算法评分 + LLM 仅生成改写建议"""
    # Step 1: 确定性评分（不依赖 LLM）
    try:
        report = evaluate(event_desc, rag_result or {}, spread_result)
    except Exception as e:
        return {"_parse_error": f"确定性评估失败：{str(e)}"}

    # Step 2: LLM 仅负责文案改写建议（LLM 真正的价值所在）
    fatal = report.get("fatal_focus", "")
    legal_score = report["scores"]["legal"]["score"]
    legal_reason = report["scores"]["legal"]["reason"]
    rep_score = report["scores"]["reputation"]["score"]

    rewrite_system = (
        "你是一位资深危机公关文案专家。请基于系统识别到的风险点，"
        "为当事方撰写一份姿态诚恳、完美避坑的官方回应或文案修改建议。"
        "输出纯文本（不要 JSON、不要 Markdown 代码块），不少于 80 字。"
        f"已知：法律风险评分 {legal_score}/5（{legal_reason}），"
        f"声誉风险评分 {rep_score}/5。"
    )
    rewrite_user = (
        f"事件背景：{event_desc}\n"
        f"视觉雷点：{visual_risk_desc or '无'}\n"
        f"被攻击焦点：{fatal}\n"
        f"原始公关草稿（需改写）：{pr_draft or '无'}\n"
        "请输出改写后的安全文案："
    )
    try:
        resp = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME,
            messages=[
                {"role": "system", "content": rewrite_system},
                {"role": "user", "content": rewrite_user},
            ],
            temperature=TEMP_REPORT,
            max_tokens=800,
        )
        rewrite = (resp.choices[0].message.content if resp.choices else "") or ""
        rewrite = rewrite.strip().replace("```", "")
    except Exception:
        rewrite = "（改写建议生成失败，请稍后重试）"

    report["rewrite_suggestion"] = rewrite
    return report


def main():
    st.set_page_config(page_title="智能文案审查系统", layout="wide")
    st.markdown(
        """
<style>
  /* 全局办公平台风格：浅灰底 + 白色内容卡片 */
  .stApp { background: #f2f4f7; color: #111827; }
  div[data-testid="stAppViewContainer"]{ background: #f2f4f7; }
  div[data-testid="stHeader"]{ background: rgba(242,244,247,0.85); }

  /* 强制浅色主题观感：侧边栏浅灰，主区域白 */
  div[data-testid="stSidebar"] > div{
    background: #f2f4f7 !important;
    color: #111827 !important;
  }
  div[data-testid="stSidebar"] *{
    color: #111827 !important;
  }
  section.main{
    color: #111827 !important;
  }
  section.main > div.block-container{
    background: #ffffff;
    border-radius: 14px;
    padding: 22px 26px 28px 26px;
    margin-top: 10px;
    margin-bottom: 28px;
    box-shadow: 0 10px 30px rgba(16,24,40,0.08);
    border: 1px solid #eef2f6;
  }
  /* 主区域大标题：强制深灰/黑色，靠上一点 */
  section.main h1,
  section.main h1 span,
  section.main [data-testid="stMarkdownContainer"] h1,
  section.main [data-testid="stMarkdownContainer"] h1 span{
    color: #333333 !important;
    margin-top: 0.2rem !important;
    margin-bottom: 0.6rem !important;
    text-shadow: none !important;
  }

  /* 统一浅色输入控件：避免深色主题残留导致文字不清晰 */
  div[data-testid="stSidebar"] textarea,
  div[data-testid="stSidebar"] input,
  div[data-testid="stSidebar"] [data-baseweb="textarea"],
  div[data-testid="stSidebar"] [data-baseweb="input"]{
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d0d5dd !important;
  }
  div[data-testid="stSidebar"] textarea::placeholder,
  div[data-testid="stSidebar"] input::placeholder{
    color: #667085 !important;
    opacity: 1 !important;
  }
  div[data-testid="stFileUploader"]{
    background: #ffffff !important;
    border: 1px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 10px 12px !important;
  }
  /* Tabs 文案/边框更清晰 */
  button[role="tab"]{
    color: #344054 !important;
  }
  button[role="tab"][aria-selected="true"]{
    color: #0D6EFD !important;
  }
  /* 让区块之间更松弛一点 */
  section.main [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]){
    margin-bottom: 10px;
  }

  /* 主按钮：企业深蓝、居中、显眼 */
  div[data-testid="main-action"] div[data-testid="stButton"] > button[kind="primary"]{
    background: #0D6EFD;
    border: 1px solid #0D6EFD;
    color: #ffffff;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.7rem 1.1rem;
    box-shadow: 0 12px 30px rgba(13,110,253,0.25);
  }
  div[data-testid="main-action"] div[data-testid="stButton"] > button[kind="primary"]:hover{
    background: #0b5ed7;
    border-color: #0b5ed7;
  }
</style>
""",
        unsafe_allow_html=True,
    )

    st.title("智能文案审查系统")

    # 主操作按钮放到页面中间最显眼的位置
    _c1, c2, _c3 = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<div data-testid="main-action">', unsafe_allow_html=True)
        run_btn = st.button("开始审查", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("输入区")
        event_desc = st.text_area(
            "事件描述",
            height=140,
            placeholder="请在这里输入你需要审查的文案，或者上传相关的图片和视频材料",
        )
        network_mood = st.text_area(
            "网络情绪",
            height=120,
            placeholder="例：愤怒、讽刺、站队明显、扩散加速…",
        )
        pr_draft = st.text_area(
            "公关草稿",
            height=180,
            placeholder="例：我们关注到相关反馈，正在核查…",
        )
        hot_topics = st.text_area(
            "🌐 当前互联网敏感热点 (选填，用逗号分隔)",
            height=72,
            placeholder="例如：姜萍学术造假事件、某企业996猝死事件...",
        )
        uploaded_file = st.file_uploader(
            "📸/🎥 上传涉事公关物料 (图片或短视频)",
            type=["jpg", "jpeg", "png", "mp4", "mov"],
        )
        if uploaded_file is not None:
            mime_type = uploaded_file.type or ""
            if mime_type.startswith("image"):
                st.image(uploaded_file, caption="已上传图片预览", use_container_width=True)
            elif mime_type.startswith("video"):
                st.video(uploaded_file)
                st.warning(
                    "⚠️ 视频转码分析中，为保证 API 稳定性，建议上传 15 秒以内的短视频片段。"
                )

    tab1, tab2, tab3 = st.tabs(["🧭 审查工作流", "📊 风险评估报告", "🗂️ 相似案例参考"])

    if run_btn:
        prog = st.progress(0, text="提取内容特征中...")
        has_text_input = bool(
            event_desc.strip() or network_mood.strip() or pr_draft.strip()
        )
        has_file_input = uploaded_file is not None
        if not (has_text_input or has_file_input):
            st.warning("请至少填写文本信息或上传图片后再启动。")
            return

        auto_generated = False
        final_event_desc = event_desc.strip()
        final_network_mood = network_mood.strip()
        final_pr_draft = pr_draft.strip()
        visual_risk_desc = ""

        prog.progress(20, text="提取内容特征中...")
        if has_file_input and (not final_event_desc) and (not final_pr_draft):
            auto_generated = True
            auto_ctx = auto_generate_context_from_image(uploaded_file)
            final_event_desc = auto_ctx.get("auto_event_desc", "").strip()
            final_network_mood = auto_ctx.get("auto_sentiment", "").strip()
            visual_risk_desc = auto_ctx.get("visual_risks", "").strip()
        elif has_file_input:
            visual_risk_desc = analyze_visual_risk(uploaded_file).strip()

        event_desc_enhanced = final_event_desc or "网络出现了一起争议事件，舆情迅速升温。"
        if visual_risk_desc.strip():
            event_desc_enhanced = (
                f"{event_desc_enhanced}\n"
                f"【视觉雷点情报】{visual_risk_desc}\n"
                "请将以上视觉雷点视为事件核心导火索之一。"
            )

        prog.progress(45, text="比对历史案例中...")
        zeitgeist_result = analyze_macro_zeitgeist(
            event_desc_enhanced, visual_risk_desc, hot_topics or ""
        )

        matched_case = retrieve_similar_case(event_desc_enhanced)

        # RAG 多库联合检索 (v2.0: 分类框架 + 关键词匹配 + 向量检索)
        try:
            rag = get_rag_engine()
            rag_result = rag.retrieve_v2(event_desc_enhanced)
        except Exception:
            rag_result = {"red_lines": [], "black_words": [], "hate_comments": [],
                          "similar_cases": [], "pattern_matches": [], "template_matches": [],
                          "audit_checklist": []}

        prog.progress(70, text="综合评估风险中...")
        with tab1:
            st.subheader("舆情动态级联沙盘")
            if matched_case:
                st.success(
                    "📚 触发历史记忆：系统检测到当前事件与【"
                    f"{matched_case.get('title', '历史案例')}"
                    "】高度相似！已将历史网民情绪与官方应对策略注入沙盘底层逻辑。"
                )
            if zeitgeist_result.get("risk_level") == "高":
                st.warning(
                    f"📡 宏观环境警告：{zeitgeist_result.get('collateral_damage_warning', '')}"
                )
            if auto_generated:
                st.info(
                    "👁️ 系统已自动从视觉物料中提取事件背景与视觉雷点..."
                    f"\n\n自动事件背景：{final_event_desc}"
                    f"\n\n自动网络情绪：{final_network_mood}"
                    f"\n\n视觉雷点：{visual_risk_desc}"
                )
            if visual_risk_desc.strip():
                st.warning(f"视觉情报局雷点：{visual_risk_desc}")

            seed_pack = generate_seed_roster(
                event_desc_enhanced, final_network_mood, final_pr_draft, rag_result=rag_result
            )
            master_reasoning = (
                seed_pack.get("master_agent_reasoning", {})
                if isinstance(seed_pack, dict)
                else {}
            )
            agents = (
                seed_pack.get("roster", [])
                if isinstance(seed_pack, dict)
                else []
            )

            st.markdown("### 🧠 Master Agent 逻辑自证中枢")
            if master_reasoning and any(master_reasoning.values()):
                render_cyber_graph(master_reasoning)
            else:
                st.info("Master Agent 已根据事件描述直接生成角色阵容（简化模式）")
            st.markdown("#### Seed Roster（种子智能体）")
            st.json(agents)
            logs = run_dynamic_sandbox(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                agents,
                zeitgeist_result=zeitgeist_result,
                matched_case=matched_case,
                rag_result=rag_result,
            )

            # 传播动力学模拟
            st.markdown("---")
            st.markdown("### 📈 SIR 传播动力学预测")
            params = map_agents_to_model_params(agents)
            spread_result = simulate(params, rounds=4)
            render_spread_chart(spread_result)

            # 展示 RAG 检索结果
            rag_ctx = get_rag_engine().format_context_for_agent(rag_result)
            if rag_ctx.strip():
                with st.expander("🔍 RAG 知识库检索结果", expanded=False):
                    st.markdown(rag_ctx)

        prog.progress(90, text="生成审查报告...")
        with tab2:
            st.subheader("舆情体检报告")
            report = generate_report(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                visual_risk_desc,
                logs,
                rag_result=rag_result,
                spread_result=spread_result,
            )
            if report.get("_parse_error"):
                st.error(f"❌ 体检报告解析失败：{report.get('_parse_error')}")
                raw_text = report.get("_raw", "")
                if raw_text:
                    st.code(raw_text)
            else:
                fatal_focus = report.get("fatal_focus", "未识别到明确集火点。")
                scores = report.get("scores", {})
                legal = scores.get("legal", {"score": "?", "reason": "暂无说明"})
                business = scores.get("business", {"score": "?", "reason": "暂无说明"})
                reputation = scores.get("reputation", {"score": "?", "reason": "暂无说明"})
                rewrite_suggestion = report.get(
                    "rewrite_suggestion", "暂无建议稿，请稍后重试。"
                )

                st.error(f"💥 被攻击焦点 / 视觉雷点：\n\n> {fatal_focus}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f"### LEGAL\n# {legal.get('score', '?')}\n"
                        f"<span style='color:gray;font-size:14px'>↑ {legal.get('reason', '暂无说明')}</span>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f"### BUSINESS\n# {business.get('score', '?')}\n"
                        f"<span style='color:gray;font-size:14px'>↑ {business.get('reason', '暂无说明')}</span>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    st.markdown(
                        f"### REPUTATION\n# {reputation.get('score', '?')}\n"
                        f"<span style='color:gray;font-size:14px'>↑ {reputation.get('reason', '暂无说明')}</span>",
                        unsafe_allow_html=True,
                    )

                st.success(f"📝 安全优化建议稿：\n\n{rewrite_suggestion}")

                save_test_record(
                    event_desc_enhanced,
                    visual_risk_desc,
                    logs,
                    report,
                )
        prog.progress(100, text="生成审查报告...")
    else:
        with tab1:
            st.info("请先在左侧输入信息，再点击上方“开始审查”。")
        with tab2:
            st.info("推演完成后，这里会展示体检报告。")

    with tab3:
        st.subheader("🗂️ 历史测试档案")

        hist_path = _history_file_path()
        if not os.path.exists(hist_path):
            st.info("暂无历史测试记录。运行审查后会自动保存。")
        else:
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = None

            if not isinstance(records, list) or len(records) == 0:
                st.info("暂无历史测试记录。运行审查后会自动保存。")
            else:
                n = len(records)

                # ---- 统计仪表盘 ----
                all_legal = []
                all_biz = []
                all_rep = []
                for rec in records:
                    rpt = rec.get("report_json") or {}
                    scores = rpt.get("scores") or {}
                    ls = scores.get("legal", {}).get("score")
                    bs = scores.get("business", {}).get("score")
                    rs = scores.get("reputation", {}).get("score")
                    if isinstance(ls, (int, float)):
                        all_legal.append(ls)
                    if isinstance(bs, (int, float)):
                        all_biz.append(bs)
                    if isinstance(rs, (int, float)):
                        all_rep.append(rs)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("总测试数", n)
                with c2:
                    avg_legal = round(sum(all_legal) / max(len(all_legal), 1), 1)
                    st.metric("平均法律风险", f"{avg_legal}/5")
                with c3:
                    avg_rep = round(sum(all_rep) / max(len(all_rep), 1), 1)
                    st.metric("平均声誉风险", f"{avg_rep}/5")
                with c4:
                    latest_ts = records[-1].get("timestamp", "—")[:10] if records else "—"
                    st.metric("最近测试", latest_ts)

                # ---- 搜索 & 筛选 ----
                st.markdown("---")
                search_col1, search_col2 = st.columns([3, 1])
                with search_col1:
                    search_term = st.text_input(
                        "🔍 搜索历史记录",
                        placeholder="输入事件关键词、危机类型或日期搜索...",
                        key="hist_search",
                    )
                with search_col2:
                    sort_order = st.selectbox(
                        "排序",
                        ["最新优先", "最早优先", "法律风险最高", "声誉风险最高"],
                        key="hist_sort",
                    )

                # ---- 过滤 ----
                filtered = []
                for i, rec in enumerate(records):
                    if search_term:
                        ed = str(rec.get("event_desc", "") or "")
                        rpt = rec.get("report_json") or {}
                        fatal = str(rpt.get("fatal_focus", "") or "")
                        ts = str(rec.get("timestamp", "") or "")
                        combined = ed + fatal + ts
                        if search_term.lower() not in combined.lower():
                            continue
                    filtered.append((i, rec))

                # ---- 排序 ----
                if sort_order == "最早优先":
                    filtered = list(reversed(filtered))
                elif sort_order == "法律风险最高":
                    filtered.sort(
                        key=lambda x: (x[1].get("report_json") or {}).get("scores", {}).get("legal", {}).get("score", 0),
                        reverse=True,
                    )
                elif sort_order == "声誉风险最高":
                    filtered.sort(
                        key=lambda x: (x[1].get("report_json") or {}).get("scores", {}).get("reputation", {}).get("score", 0),
                        reverse=True,
                    )
                # 默认: 最新优先 (records 已经是最新在后的顺序，filtered 保持了原始顺序即最早在前 → 需要反转)

                st.caption(f"共 {n} 条记录，当前显示 {len(filtered)} 条")

                # ---- 记录列表 ----
                for display_idx, (actual_idx, rec) in enumerate(filtered):
                    rpt = rec.get("report_json") or {}
                    ed = str(rec.get("event_desc", "") or "")
                    fatal = str(rpt.get("fatal_focus", "") or "")
                    ts = str(rec.get("timestamp", "") or "")

                    # 截取预览
                    case_preview = ed[:40] + "…" if len(ed) > 40 else (ed or "无描述")
                    fatal_preview = fatal[:20] + "…" if len(fatal) > 20 else (fatal or "无")

                    # 风险评分徽章
                    scores = rpt.get("scores") or {}
                    leg_s = scores.get("legal", {}).get("score", "?")
                    biz_s = scores.get("business", {}).get("score", "?")
                    rep_s = scores.get("reputation", {}).get("score", "?")

                    def _risk_badge(score):
                        if isinstance(score, (int, float)):
                            if score >= 4:
                                return f"🔴 {score}"
                            elif score >= 3:
                                return f"🟠 {score}"
                            elif score >= 2:
                                return f"🟡 {score}"
                            return f"🟢 {score}"
                        return f"⚪ {score}"

                    title = (
                        f"{_risk_badge(rep_s)} 声誉 | "
                        f"{_risk_badge(leg_s)} 法律 | "
                        f"{_risk_badge(biz_s)} 商业  —  "
                        f"🗂️ {case_preview}"
                    )

                    with st.expander(title):
                        # 元数据行
                        st.caption(f"📅 {ts} ｜ 🎯 致命点：{fatal_preview}")

                        # 事件背景
                        st.markdown("**事件背景**")
                        st.text(ed[:500] + ("…" if len(ed) > 500 else ""))

                        # 视觉雷点
                        vr = (rec.get("visual_risks") or "").strip()
                        if vr:
                            st.markdown("**视觉雷点**")
                            st.text(vr[:300])

                        # 三维评分
                        col_s1, col_s2, col_s3 = st.columns(3)
                        with col_s1:
                            st.metric("LEGAL", leg_s)
                        with col_s2:
                            st.metric("BUSINESS", biz_s)
                        with col_s3:
                            st.metric("REPUTATION", rep_s)

                        # 优化建议
                        rewrite = rpt.get("rewrite_suggestion", "—")
                        if rewrite and rewrite != "—":
                            st.markdown("**优化建议**")
                            st.info(rewrite[:400])

                        # 操作按钮行
                        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                        with btn_col1:
                            if st.button("🗑️ 删除", key=f"del_hist_{actual_idx}"):
                                records.pop(actual_idx)
                                with open(hist_path, "w", encoding="utf-8") as f:
                                    json.dump(records, f, ensure_ascii=False, indent=2)
                                st.rerun()
                        with btn_col2:
                            if st.button("📋 添加到知识库", key=f"promote_{actual_idx}"):
                                # 将本次测试转化为知识库案例
                                new_case = {
                                    "id": f"case_promoted_{ts[:10]}_{actual_idx}",
                                    "title": case_preview.replace("…", ""),
                                    "crisis_type": fatal_preview.replace("…", "") or "用户测试案例",
                                    "description": ed[:300],
                                    "public_reaction": f"模拟推演结果 — 法律风险:{leg_s}/5, 声誉风险:{rep_s}/5",
                                    "pr_response": str(rpt.get("rewrite_suggestion", ""))[:300],
                                    "pr_result": f"自动提升自历史测试 #{actual_idx}",
                                    "source": "user_promoted",
                                }
                                _promote_to_case_library(new_case)
                                st.success(f"✅ 已添加到案例库 (ID: {new_case['id']})")
                                st.rerun()
                        with btn_col3:
                            # 导出单条记录
                            record_json = json.dumps(rec, ensure_ascii=False, indent=2)
                            st.download_button(
                                "💾 导出",
                                data=record_json,
                                file_name=f"pr_audit_{ts[:10]}_{actual_idx}.json",
                                mime="application/json",
                                key=f"export_{actual_idx}",
                            )

                # ---- 批量操作 ----
                st.markdown("---")
                bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
                with bulk_col1:
                    if st.button("🗑️ 清空全部记录", type="secondary", key="clear_all_hist"):
                        st.session_state["confirm_clear"] = True
                with bulk_col2:
                    all_json = json.dumps(records, ensure_ascii=False, indent=2)
                    st.download_button(
                        "💾 导出全部记录 (JSON)",
                        data=all_json,
                        file_name=f"pr_audit_all_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json",
                        key="export_all",
                    )
                with bulk_col3:
                    # CSV 导出（精简版）
                    csv_lines = ["时间,事件摘要,法律风险,商业风险,声誉风险,致命点"]
                    for rec in records:
                        rpt = rec.get("report_json") or {}
                        scores = rpt.get("scores") or {}
                        csv_lines.append(
                            f"\"{rec.get('timestamp','')}\","
                            f"\"{str(rec.get('event_desc',''))[:60].replace(chr(10),' ')}\","
                            f"{scores.get('legal',{}).get('score','?')},"
                            f"{scores.get('business',{}).get('score','?')},"
                            f"{scores.get('reputation',{}).get('score','?')},"
                            f"\"{str(rpt.get('fatal_focus',''))[:60]}\""
                        )
                    st.download_button(
                        "📊 导出全部记录 (CSV)",
                        data="\n".join(csv_lines),
                        file_name=f"pr_audit_all_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="export_csv",
                    )

                # 二次确认清空
                if st.session_state.get("confirm_clear"):
                    st.error("⚠️ 确认清空所有 {n} 条历史记录？此操作不可撤销！")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ 确认清空", type="primary"):
                            with open(hist_path, "w", encoding="utf-8") as f:
                                json.dump([], f)
                            st.session_state["confirm_clear"] = False
                            st.rerun()
                    with c2:
                        if st.button("❌ 取消"):
                            st.session_state["confirm_clear"] = False
                            st.rerun()


if __name__ == "__main__":
    main()
