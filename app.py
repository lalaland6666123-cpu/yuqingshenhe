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

from config import MODEL_NAME, QWEN_API_KEY, QWEN_BASE_URL, VL_MODEL_NAME


HISTORY_FILE = "history_records.json"


client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

CASE_DATABASE = []
CASE_INDEX = []


def stream_markdown_effect(container, text, prefix="", suffix="", speed=0.03):
    """模拟打字机流式输出的视觉效果"""
    placeholder = container.empty()
    displayed_text = ""
    for char in text or "":
        displayed_text += char
        # 加上一个闪烁的光标 ▌ 增加赛博科技感
        placeholder.markdown(f"{prefix}{displayed_text}▌{suffix}")
        time.sleep(speed)
    # 结束后移除光标
    placeholder.markdown(f"{prefix}{displayed_text}{suffix}")


def _load_case_study_database():
    global CASE_DATABASE, CASE_INDEX
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "case_studies.json"),
        os.path.join(os.path.dirname(base_dir), "case_studies.json"),
    ]
    CASE_DATABASE = []
    CASE_INDEX = []
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                continue
            CASE_DATABASE = [c for c in data if isinstance(c, dict)]
            CASE_INDEX = [
                {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "crisis_type": c.get("crisis_type"),
                }
                for c in CASE_DATABASE
            ]
            return
        except Exception:
            continue


_load_case_study_database()


def retrieve_similar_case(event_desc):
    if not CASE_INDEX or not event_desc.strip():
        return None
    system_prompt = (
        "你是一个资深舆情档案管理员。请分析当前的事件背景，并从提供的历史案例索引库中，"
        "找出一个在【舆情性质、网民情绪或公关难点】上最相似的案例。"
        "如果找到，请直接输出该案例的完整 ID（如 case_1）；如果没有高度相似的，输出 none。"
        "只输出一行内容：要么是形如 case_1 的 ID，要么是单词 none，不要标点、解释或 Markdown。"
    )
    user_prompt = (
        f"当前事件背景：{event_desc}\n\n"
        f"历史案例索引（JSON）：\n{json.dumps(CASE_INDEX, ensure_ascii=False)}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=32,
        )
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
    except Exception:
        return None
    raw_stripped = raw.strip()
    if not raw_stripped:
        return None
    low = raw_stripped.lower()
    if re.fullmatch(r"none\s*", low) or low.startswith("none"):
        return None
    m = re.search(r"case_[a-z0-9_]+", raw_stripped, re.I)
    if not m:
        return None
    cid = m.group(0)
    for item in CASE_DATABASE:
        if item.get("id") == cid:
            return item
    return None


MANDATORY_CHAT_RULE = (
    "最高指令：严禁使用首先其次、作为一名等书面语！"
    "必须输出20字以内的口语化弹幕！带情绪！严禁换行！"
)


def _history_file_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), HISTORY_FILE)


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
            temperature=0.9,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        out = (content or "").strip()
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
        resp = client.chat.completions.create(
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
            temperature=0.2,
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
        resp = client.chat.completions.create(
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
            temperature=0.2,
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


def _llm_json_array(system_prompt, user_prompt, max_tokens=600, temperature=0.45):
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
    keys = (
        "政务",
        "政府",
        "官员",
        "部门",
        "通报",
        "调查组",
        "纪委",
        "基层",
        "干部",
        "官方",
        "市委",
        "省委",
        "报案",
        "公文",
        "舆情处置",
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


def _default_supportive_fan_agent(event_desc):
    if _is_gov_context(event_desc):
        return {
            "name": "本地理性支持者",
            "persona": "仍愿意相信官方会把事情说清楚、呼吁先勿传谣的本地网民",
            "weight": 455,
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
        "根据事件语境，设计恰好 2 个会在第二轮引爆对线的对抗型高影响力角色，姓名与人设必须贴合当前事件，禁止套用无关模板。"
        "政务/公共事件可类似：知名时事评论员、较真考据网友；商业/消费维权可类似：硬核打假博主、维权意见领袖；娱乐热点可类似：毒舌娱乐博主、饭圈对立大粉。"
        "输出格式：严格为 JSON 数组，长度恰好为 2，每项仅含字符串字段 name、persona。禁止 Markdown、禁止解释。"
    )
    user_prompt = (
        f"事件描述：{event_desc}\n"
        f"网络情绪：{network_mood}\n"
        "请输出上述 JSON 数组。"
    )
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=400, temperature=0.55)
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
                {"name": "知名时事评论员", "persona": "擅长公共议题拆解、语气尖锐的头部评论账号"},
                {"name": "较真考据网友", "persona": "逐帧抠细节、专盯时间线与证据链的技术型网民"},
            ]
        else:
            out = [
                {"name": "硬核打假博主", "persona": "专做品牌翻车的测评与实锤合集"},
                {"name": "高活跃维权大V", "persona": "带动消费者集体投诉与话题冲榜的意见领袖"},
            ]
    return out[:2]


def generate_seed_roster(event_desc, network_mood, pr_draft):
    system_prompt = (
        "你是舆情模拟引擎，只输出 JSON。"
        "你现在是全局总控智能体(Master Agent)。在生成具体角色前，你必须先阐述你的排兵布阵逻辑。"
        "请先判断事件的性质（政务、商业、娱乐圈或社会热点），再生成 2~3 个种子智能体；角色必须与性质严格匹配。"
        "请先评估该事件的【公关防御难度】。"
        "【强制支持者规则】除「极其恶劣的政务造假/公然作秀糊弄公众」（如一眼假的官方PS通报、政务通报造假）外，"
        "无论风险多高，种子阵容中**必须恰好包含 1 名 stance 为 supportive 的角色**："
        "商业语境须为「品牌死忠粉/口味粉/长期用户」，代表危机爆发初期仍站队的一方；"
        "政务（非上述极端造假）语境可为「仍愿意等候调查、反对造谣的温和认同者」，不得使用饭圈话术。"
        "仅当事件同时满足：政务属性 + 上述极端造假/作秀性质时，才允许完全不生成 supportive，其余立场以 hostile 与少量 neutral 为主。"
        "若属有争议、有粉丝基础的商业或娱乐事件，可在「1 名 supportive 早期支持者」之外，再按常理分配 hostile/neutral，禁止凭空出现大规模洗地军团。"
        "若为政务事件：必须生成类似「当地宣传部门」「涉事单位回应人」及群众侧角色，绝对不能出现「品牌客服」等商业错位话术（但允许前述 1 名非饭圈 supportive）。"
        "若为商业事件：可生成「品牌公关」「消费者」「行业观察者」等，且须含 1 名 supportive 老粉/口味粉。"
        "若为娱乐圈或社会热点：生成符合该圈层语境的角色。"
        "每个智能体字段：name, persona, weight, stance, role_type。"
        "weight 必须是 100~1000 的整数。"
        "stance 仅可为 supportive/neutral/hostile。"
        "role_type 仅可为 official（官方/涉事主体口径）、influencer（大V或强意见领袖）、bystander（路人或围观者）。"
        "请返回如下 JSON 结构："
        '{'
        '  "master_agent_reasoning": {'
        '    "trigger_anchor": "一句话概括当前事件最核心的舆情毒点（如：视觉中发现‘像狗一样’的侮辱性词汇）",'
        '    "rag_evidence": "结合匹配到的历史案例（如：李佳琦事件），指出历史规律（如：此类傲慢会引发网民极度背刺感，产生400%负面情绪）",'
        '    "strategy_argument": "基于上述论据，解释你为什么配置接下来的这些角色阵营（如：因此本局配置80%极端敌意角色，测试系统压力阈值）",'
        '    "evolution_prediction": "用一句话预判舆情演化的最可能分支（如：大V二次剪辑扩散->品牌被扣傲慢帽子->官方通报被嘲讽甩锅）"'
        "  },"
        '  "roster": [ {现有的角色结构} ]'
        "}"
        "只允许输出上述 JSON 对象，禁止 Markdown、禁止解释。"
    )
    user_prompt = (
        f"事件描述：{event_desc}\n"
        f"网络情绪：{network_mood}\n"
        f"公关草稿：{pr_draft}\n"
        "请严格按要求输出上述 JSON 对象。"
    )
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=700, temperature=0.4)
    parsed = safe_json_loads(raw, default_value={})

    master_agent_reasoning = {}
    roster = []

    if isinstance(parsed, dict):
        master_agent_reasoning = parsed.get("master_agent_reasoning") or {}
        roster = parsed.get("roster") or []
    elif isinstance(parsed, list):
        # 兼容旧模型/波动：仍可能只返回数组
        roster = parsed

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

    if not _is_extreme_gov_fraud(event_desc) and cleaned:
        if not any(a.get("stance") == "supportive" for a in cleaned):
            fan = _default_supportive_fan_agent(event_desc)
            if len(cleaned) >= 3:
                cleaned[-1] = fan
            else:
                cleaned.append(fan)
            cleaned = cleaned[:3]

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
    trigger_anchor = str(mr.get("trigger_anchor", "") or "").strip() or "（未提供锚点）"
    rag_evidence = str(mr.get("rag_evidence", "") or "").strip() or "（未提供论据）"
    strategy_argument = (
        str(mr.get("strategy_argument", "") or "").strip() or "（未提供论点）"
    )

    st.markdown("### 🧠 Master Agent 逻辑自证中枢")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📍 **多模态语义锚点**\n\n{trigger_anchor}")
        with col2:
            st.warning(f"📚 **历史论据映射 (RAG)**\n\n{rag_evidence}")
        with col3:
            st.success(f"⚙️ **阵营推演策略 (Argument)**\n\n{strategy_argument}")

        st.markdown(
            f"""
```log
[System] 深度识别完成：定位多模态语义风险点 -> {trigger_anchor}
[RAG_Match] 触发历史记忆关联 -> {rag_evidence}
[Master_Agent] 执行角色阵营装载 -> {strategy_argument}
[Status] 逻辑自证闭环完成。沙盘推演引擎启动...
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
    raw = _llm_json_array(system_prompt, user_prompt, max_tokens=400, temperature=0.35)
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


def generate_round3_swarm_danmaku_text(event_desc, network_mood, pr_draft, logs):
    system_prompt = (
        "你是一个汇聚成千上万路人跟帖的弹幕生成器，只输出弹幕列表本身。"
        "最高指令：你现在代表成千上万的吃瓜网民和暴怒群众。"
        "请仔细阅读用户提供的沙盘对线记录与事件背景，然后一口气输出 8 到 10 条极短（每条 15 字以内）、"
        "情绪各异、立场分化的网友跟帖弹幕。"
        "不要任何多余的说明、标题或开场白；每条弹幕单独占一行；每行必须以破折号「—」开头（全角破折号）。"
        "模拟微博热搜彻底炸锅的刷屏效果。"
    )
    user_prompt = (
        f"当前热搜话题：{event_desc}\n"
        f"网络情绪底色：{network_mood}\n"
        f"公关方公开表态：{pr_draft}\n"
        f"沙盘对线日志（JSON）：{json.dumps(logs, ensure_ascii=False)}\n"
        "请立即输出 8～10 行弹幕。"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.92,
            max_tokens=500,
        )
        return (resp.choices[0].message.content if resp.choices else "") or ""
    except Exception as e:
        return f"—弹幕生成失败：{e}"


def render_round3_swarm_danmaku_ui(raw_text):
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    if not lines:
        lines = ["—（暂无弹幕）"]
    body = "\n".join(lines[:14])
    safe = html.escape(body)
    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#1c1917 0%,#450a0a 100%);border:1px solid #dc2626;
border-radius:10px;padding:14px 16px;margin:12px 0 20px 0;box-shadow:0 4px 18px rgba(220,38,38,0.25);">
<p style="margin:0 0 10px 0;font-weight:700;color:#fecaca;font-size:16px;">
🚀 【全网情绪沸腾 / 实时弹幕涌入】
</p>
<pre style="white-space:pre-wrap;font-size:12.5px;line-height:1.65;margin:0;color:#fee2e2;
font-family:system-ui,sans-serif;">{safe}</pre>
</div>
""",
        unsafe_allow_html=True,
    )


def build_plaza_context(event_desc, logs_rows):
    lines = [
        f"当前热搜话题：{event_desc}",
        "--- 实时广场热评 ---",
    ]
    if not logs_rows:
        lines.append("（广场暂无高赞摘选，本话题刚开始发酵。）")
    else:
        for i, row in enumerate(logs_rows, 1):
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


def _rag_official_memory_tail(matched_case):
    pr = str(matched_case.get("pr_response") or "")
    res = str(matched_case.get("pr_result") or "")
    return (
        "【公关错题本警告】：历史上在处理相似危机时，官方曾做出如下应对：『"
        f"{pr}"
        "』，导致了『"
        f"{res}"
        "』的后果。请你作为当前的官方，深刻反思这个失败/成功的教训。"
        "你可以选择重蹈覆辙（傲慢/甩锅），或者给出更聪明的声明！"
    )


def _rag_crowd_memory_tail(matched_case):
    pub = str(matched_case.get("public_reaction") or "")
    return (
        "【真实情绪参考】：在历史上极其相似的事件中，真实网民的反应是：『"
        f"{pub}"
        "』。请你吸收这种真实存在的愤怒点、痛点和阴阳怪气的切入点，结合当前事件狠狠地发帖！"
    )


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
):
    logs = []
    primary_catalyst_name = ""
    catalyst_names = set()

    for round_idx in range(1, 5):
        st.markdown(f"### Round {round_idx}")

        if round_idx == 2:
            catalyst_specs = generate_dynamic_catalyst_agents(event_desc, network_mood)
            injected = []
            for i, spec in enumerate(catalyst_specs):
                injected.append(
                    {
                        "name": spec["name"],
                        "persona": spec["persona"],
                        "weight": 500000 if i == 0 else 100000,
                        "stance": "hostile",
                        "role_type": "influencer",
                    }
                )
            agents.extend(injected)
            if injected:
                primary_catalyst_name = injected[0]["name"]
                catalyst_names = {a["name"] for a in injected}
            st.error(
                "🚨 警告：检测到敏感词！触发舆情破圈！"
                f"对立节点「{injected[0]['name']}」「{injected[1]['name']}」已空降战场！"
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
                rag_tail = (
                    _rag_official_memory_tail(matched_case)
                    if matched_case
                    else ""
                )
                system_prompt = (
                    f"你执笔对外口径的官方身份：{agent['name']}（{agent['persona']}）。"
                    "你只负责机构通报文本，不参与网民互怼，也不使用聊天语气。"
                    f"{OFFICIAL_OUTPUT_FORMAT_HARD}"
                    f"{macro_bg}"
                    f"{rag_tail}"
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
                    max_tokens=400,
                    skip_mandatory_chat_rule=True,
                    collapse_newlines=False,
                )
                phase = "初步定调" if round_idx == 1 else "最终通报"
                render_official_announcement(agent, speech, round_idx, phase)
            else:
                spiral_snippet = ""
                if agent.get("stance") == "supportive":
                    if round_idx == 1:
                        spiral_snippet = (
                            "【沉默的螺旋·Round1】你是品牌的死忠粉或长期支持者，你认为这只是一点小问题或小误会，"
                            "请理直气壮地为品牌/当事方辩护（例如：只要好吃/好用就行，管那么多干嘛、别上纲上线）。"
                        )
                    elif round_idx == 2:
                        spiral_snippet = (
                            "【沉默的螺旋·Round2】大V已经下场带节奏，全网都在骂。你的底气开始不足，"
                            "语气变得犹豫，试图和稀泥或转移话题，不再像上一轮那样硬气。"
                        )
                    elif round_idx == 3:
                        spiral_snippet = (
                            "【沉默的螺旋·Round3】舆论已经彻底失控。你感到害怕和无力，作为曾经的支持者，"
                            "发出一句失望的叹息，或表示不想再管了、心累了。"
                        )
                rag_crowd = ""
                if matched_case and rt_agent in ("bystander", "influencer"):
                    rag_crowd = _rag_crowd_memory_tail(matched_case)
                round4_law = ROUND4_FINAL_BEHAVIOR_LAW if round_idx == 4 else ""
                system_prompt = (
                    f"账号人设：{agent['name']}。"
                    f"人设摘要：{agent['persona']}。"
                    f"立场倾向：{agent['stance']}。"
                    f"传播权重：{agent['weight']}。"
                    f"身份类型：{rt_agent}（非机构通报账号）。"
                    "你在微博类公开广场的话题链路下独立发帖，不是微信群聊，也不是私聊。"
                    f"{macro_bg}"
                    f"{spiral_snippet}"
                    f"{round4_law}"
                    "发言须符合人设，与同话题下其他贴文风避免雷同套话。"
                    f"{rag_crowd}"
                )
                round_instruction = "请发布一条广场动态（单条，勿换行堆砌长文）。"
                if round_idx == 1:
                    round_instruction = "Round1：试探性带话题发帖，观察广场风向。"
                elif round_idx == 2:
                    round_instruction = "Round2：话题热度抬升，请强化情绪与观点输出，仍遵守广场发帖原则。"
                elif round_idx == 3:
                    round_instruction = "Round3：焦灼对撞期，围绕话题与公共证据链输出，不点名吵架。"
                elif round_idx == 4:
                    round_instruction = "Round4：结局回合，请按上文最高指令完成最后一条广场动态。"

                anti_echo = ""
                if current_round_responses:
                    joined = "\n".join(current_round_responses)
                    anti_echo = (
                        "\n【最高优先级警告】以下是刚才别人发的帖子（本回合内已出现）：\n"
                        f"{joined}\n"
                        "你绝对不能重复他们的句式和核心词！你必须找一个全新的切入点！"
                        "例如：别人骂态度，你就骂价格；别人骂质量，你就谈知情权；别人骂品牌，你就谈童年滤镜破灭。"
                        "请输出极具个人特色的独立观点！"
                    )

                user_prompt = (
                    f"{context_text}\n"
                    f"网络情绪：{network_mood}\n"
                    f"涉事方公开表态参考（可引用嘲讽但不要假装私聊回复对方）：{pr_draft}\n"
                    f"{round_instruction}\n"
                    f"{PLAZA_SQUARE_RULE}"
                    f"{anti_echo}"
                )
                if round_idx == 4:
                    user_prompt += "\n" + ROUND4_ANTI_COLLISION_TAIL
                speech = chat_llm(system_prompt, user_prompt, max_tokens=80)
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
            current_round_responses.append(f"@{agent['name']}：{speech}")

        if round_idx == 3:
            swarm_raw = generate_round3_swarm_danmaku_text(
                event_desc, network_mood, pr_draft, logs
            )
            render_round3_swarm_danmaku_ui(swarm_raw)

    return logs


def generate_report(event_desc, network_mood, pr_draft, visual_risk_desc, logs):
    system_prompt = (
        "你现在是年薪百万的顶级危机公关总监。请审阅刚才的舆情沙盘推演日志，并给出结构化的定损报告。"
        "请直接输出合法的 JSON，格式如下："
        "{"
        '"fatal_focus":"一针见血地指出网民集火攻击的最核心点（是某句傲慢的文案，还是某个糟糕的视觉细节）",'
        '"scores":{'
        '"legal":{"score":1到5的整数,"reason":"简短说明法律风险"},'
        '"business":{"score":1到5的整数,"reason":"简短说明商业/销量影响"},'
        '"reputation":{"score":1到5的整数,"reason":"简短说明品牌声誉损毁度"}'
        "},"
        '"rewrite_suggestion":"基于以上雷点，给出一份姿态诚恳、完美避坑的官方回应或文案修改建议（不少于100字）。"'
        "}"
    )
    user_prompt = (
        f"事件描述：{event_desc}\n"
        f"视觉雷点：{visual_risk_desc or '无'}\n"
        f"网络情绪：{network_mood}\n"
        f"公关草稿：{pr_draft}\n"
        f"完整4轮沙盘日志：{json.dumps(logs, ensure_ascii=False)}"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
        )
        raw = resp.choices[0].message.content if resp.choices else ""
    except Exception as e:
        return {"_parse_error": f"报告生成失败：{str(e)}"}

    cleaned = (raw or "").replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("报告不是 JSON 对象")
        if "fatal_focus" not in parsed or "scores" not in parsed or "rewrite_suggestion" not in parsed:
            raise ValueError("报告字段缺失")
        return parsed
    except Exception as e:
        return {"_parse_error": f"报告 JSON 解析失败：{str(e)}", "_raw": cleaned[:800]}


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
                event_desc_enhanced, final_network_mood, final_pr_draft
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
            render_cyber_graph(master_reasoning or {})
            st.markdown("#### Seed Roster（种子智能体）")
            st.json(agents)
            logs = run_dynamic_sandbox(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                agents,
                zeitgeist_result=zeitgeist_result,
                matched_case=matched_case,
            )

        prog.progress(90, text="生成审查报告...")
        with tab2:
            st.subheader("舆情体检报告")
            report = generate_report(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                visual_risk_desc,
                logs,
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
            st.info("暂无历史测试记录。")
        else:
            try:
                with open(hist_path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception:
                records = None
            if not isinstance(records, list) or len(records) == 0:
                st.info("暂无历史测试记录。")
            else:
                n = len(records)
                for display_idx, rec in enumerate(reversed(records)):
                    actual_idx = n - 1 - display_idx
                    rpt = rec.get("report_json") or {}
                    ed = str(rec.get("event_desc", "") or "")
                    fatal = str(rpt.get("fatal_focus", "") or "")
                    case_preview = ed[:20] + "…" if len(ed) > 20 else (ed or "无")
                    fatal_preview = fatal[:10] + "…" if len(fatal) > 10 else (fatal or "无")
                    title = f"🗂️ 案例：{case_preview} | 致命点：{fatal_preview}"
                    with st.expander(title):
                        st.markdown(f"**事件背景**\n\n{rec.get('event_desc', '—')}")
                        vr = (rec.get("visual_risks") or "").strip()
                        if vr:
                            st.markdown(f"**视觉雷点**\n\n{vr}")
                        else:
                            st.markdown("**视觉雷点**\n\n（无）")
                        scores = rpt.get("scores") or {}
                        leg = scores.get("legal", {})
                        bus = scores.get("business", {})
                        rep = scores.get("reputation", {})
                        st.markdown(
                            "**风险定损**\n\n"
                            f"- LEGAL：**{leg.get('score', '?')}**\n"
                            f"- BUSINESS：**{bus.get('score', '?')}**\n"
                            f"- REPUTATION：**{rep.get('score', '?')}**"
                        )
                        st.markdown(
                            f"**优化建议**\n\n{rpt.get('rewrite_suggestion', '—')}"
                        )
                        if st.button(
                            "🗑️ 删除此记录",
                            key=f"del_hist_{actual_idx}",
                        ):
                            records.pop(actual_idx)
                            with open(hist_path, "w", encoding="utf-8") as f:
                                json.dump(records, f, ensure_ascii=False, indent=2)
                            st.rerun()


if __name__ == "__main__":
    main()
