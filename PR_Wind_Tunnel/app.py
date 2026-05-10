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

# 种子阵容人数上限；过小沙盘冷清，过大则每轮全员发言的 LLM 调用与耗时显著增加。
SEED_ROSTER_CAP = 14

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


def _verify_risk_quality(risk_text, event_desc=""):
    """自检核心雷点质量：评估是否足够精准、尖锐、具有代表性"""
    system_prompt = (
        "你是公关质检员。请对给定的公关雷点分析进行质量评估。"
        "评估标准：1）是否精准定位到最致命的雷点（而非泛泛而谈）；2）是否一针见血、言辞犀利；3）是否具体到涉事主体的某个具体行为/话语/视觉细节。"
        "只输出 JSON：{\"score\": 1-5的整数, \"reason\": \"评分理由\", \"need_retry\": true或false}"
    )
    user_prompt = (
        f"事件背景：{event_desc or '无'}\n\n"
        f"待评估的雷点分析：{risk_text}\n\n"
        "请严格评估质量。如果得分低于4分或需要重试，请指出不足。"
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return parsed
    except Exception:
        return {"score": 3, "reason": "评估失败，默认需要重试", "need_retry": True}


def analyze_visual_risk(uploaded_file, event_desc="", max_retries=2):
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

    # 如果有事件描述，将其加入 prompt
    base_prompt = ANALYZE_VISUAL_RISK_PROMPT
    if event_desc and event_desc.strip():
        base_prompt = (
            f"【事件背景描述】：{event_desc.strip()}\n\n"
            "请结合以上背景，审视下方视觉物料，分析其中可能存在的公关雷点。\n\n"
            + ANALYZE_VISUAL_RISK_PROMPT
        )

    last_risk = ""
    for attempt in range(max_retries + 1):
        # 如果是重试，增加质检反馈到 prompt 中
        current_prompt = base_prompt
        if attempt > 0 and last_risk:
            quality_check = _verify_risk_quality(last_risk, event_desc)
            if not quality_check.get("need_retry", True):
                # 质量合格，直接返回
                return last_risk
            feedback = quality_check.get("reason", "分析不够精准")
            current_prompt = (
                f"【上一次分析质量评估反馈】：{feedback}\n\n"
                "请根据以上反馈，重新审视视觉物料，给出更精准、更尖锐的核心雷点分析。\n\n"
                + base_prompt
            )

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
                                "text": current_prompt,
                            },
                        ],
                    },
                ],
                temperature=0.2 if attempt == 0 else 0.3,
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
                    last_risk = str(parsed.get("visual_risks", "")).strip()
                    if last_risk:
                        continue  # 进入下一次质检循环
            except Exception:
                pass
            last_risk = raw_content.replace("\n", " ")
        except Exception as e:
            if attempt == 0:
                return f"视觉分析失败：{e}"
            # 重试失败，返回上一次结果
            break

    # 所有重试结束后，返回最后一次结果
    return last_risk if last_risk else "视觉分析未返回有效结果"


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
    "【广场发帖原则】：你是在自己的社交账号上公开发帖。"
    "严禁点对点回复特定的人（严禁使用「某某说得对」「你这是狡辩」等吵架句式）！"
    "【绝对格式禁令】：严禁在文案开头或结尾加上任何类似 #xxx# 的话题标签！绝对不允许！直接说正文内容！"
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
        "请先判断事件的性质（政务、商业、娱乐圈或社会热点），再生成 10~14 个种子智能体；角色必须与性质严格匹配。"
        "阵容须呈现「全网热议、多方混战」：覆盖锐评大V、路透搬运、本地围观、反讽段子手、科普考据、对立阵营意见领袖、行业观察、营销号带节奏等多路网民画像，立场与权重拉开梯度，禁止几张嘴同质化复读。"
        "【乐子人强制规则】：你的名单中**必须包含 2~3 名「网络乐子人/反串黑段子手/阴阳怪气网民」**，他们看热闹不嫌事大，专门喜欢改编官方话术。"
        "请先评估该事件的【公关防御难度】（低/中/高/极高），并说明判断依据（如涉事主体过往公信力、事件敏感度、视觉物料冲击力等）。"
        "【强制支持者规则】除「极其恶劣的政务造假/公然作秀糊弄公众」（如一眼假的官方PS通报、政务通报造假）外，"
        "无论风险多高，种子阵容中**必须至少包含 1 名 stance 为 supportive 的角色**："
        "商业语境须为「品牌死忠粉/口味粉/长期用户」，代表危机爆发初期仍站队的一方；"
        "政务（非上述极端造假）语境可为「仍愿意等候调查、反对造谣的温和认同者」，不得使用饭圈话术。"
        "每个智能体字段：name, persona, weight, stance, role_type。"
        "weight 必须是 100~1000 的整数，代表该角色在舆情场中的声量权重。"
        "stance 仅可为 supportive/neutral/hostile。"
        "role_type 仅可为 official（官方/涉事主体口径）、influencer（大V或强意见领袖）、bystander（路人或围观者）。"
        "请返回如下 JSON 结构："
        "{"
        '  "master_agent_reasoning": {'
        '    "trigger_anchor": "一句话精准概括当前事件最核心的舆情引爆点（必须具体到涉事主体的某个具体行为/话语/视觉细节）",'
        '    "crisis_nature": "明确判定事件性质（政务/商业/娱乐/社会），并简述判定依据",'
        '    "defense_difficulty": "评估公关防御难度（低/中/高/极高），并给出1-2句判断依据",'
        '    "rag_evidence": "若匹配到历史案例，指出该案例的核心教训；若无匹配，说明本次为新型/独立事件",'
        '    "strategy_argument": "详细阐述你的排兵布阵逻辑：为什么配置这些角色？各阵营如何相互作用推动舆情演化？重点指出哪些角色会形成对线、哪些角色会带节奏、哪些角色会反转局势。不少于50字。",'
        '    "evolution_prediction": "用一句话预判舆情演化的最可能分支（如：官方强硬回应→网民更愤怒→倒逼第三方介入；或官方诚恳道歉+赔偿→情绪缓和→部分网民转为支持）"'
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
    raw = _llm_json_array(
        system_prompt, user_prompt, max_tokens=3200, temperature=0.6
    )
    parsed = safe_json_loads(raw, default_value={})

    master_agent_reasoning = {}
    roster = []

    if isinstance(parsed, dict):
        master_agent_reasoning = parsed.get("master_agent_reasoning") or {}
        roster = parsed.get("roster") or []
    elif isinstance(parsed, list):
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
            cleaned.append(fan)
            cleaned = cleaned[:SEED_ROSTER_CAP]

    if len(cleaned) < 5:
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
                {
                    "name": "阴阳怪气乐子人",
                    "persona": "擅长提炼官方通报里的病句和雷人词汇进行造句模仿的段子手",
                    "weight": 400,
                    "stance": "hostile",
                    "role_type": "bystander",
                },
                {
                    "name": "愤怒的围观群众",
                    "persona": "对官方极度不信任、情绪暴躁的路人",
                    "weight": 550,
                    "stance": "hostile",
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
                    "name": "维权消费者",
                    "persona": "因产品或服务体验受损极其愤怒的用户",
                    "weight": 610,
                    "stance": "hostile",
                    "role_type": "bystander",
                },
                {
                    "name": "十年老粉用户",
                    "persona": "长期复购，遇事本能想先替品牌说两句",
                    "weight": 500,
                    "stance": "supportive",
                    "role_type": "bystander",
                },
                {
                    "name": "反串黑段子手",
                    "persona": "看热闹不嫌事大，最喜欢把官方的公关原话编成段子、做成梗图嘲讽",
                    "weight": 550,
                    "stance": "hostile",
                    "role_type": "bystander",
                },
                {
                    "name": "行业理中客大V",
                    "persona": "喜欢深度拆解商业逻辑，语气带着爹味和优越感",
                    "weight": 580,
                    "stance": "neutral",
                    "role_type": "influencer",
                },
            ]

    mr = master_agent_reasoning if isinstance(master_agent_reasoning, dict) else {}
    return {
        "master_agent_reasoning": {
            "trigger_anchor": str(mr.get("trigger_anchor", "") or "").strip(),
            "crisis_nature": str(mr.get("crisis_nature", "") or "").strip(),
            "defense_difficulty": str(mr.get("defense_difficulty", "") or "").strip(),
            "rag_evidence": str(mr.get("rag_evidence", "") or "").strip(),
            "strategy_argument": str(mr.get("strategy_argument", "") or "").strip(),
            "evolution_prediction": str(mr.get("evolution_prediction", "") or "").strip(),
        },
        "roster": cleaned[:SEED_ROSTER_CAP],
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
border-radius:10px;padding:14px 16px;margin:12px 0 20px 0;box-shadow:0 4px 18px rgba(220,38,38,0.25);color:#ffffff;">
<p style="margin:0 0 10px 0;font-weight:700;color:#ffffff !important;font-size:16px;">
🚀 【全网情绪沸腾 / 实时弹幕涌入】
</p>
<div style="white-space:pre-wrap;font-size:12.5px;line-height:1.65;margin:0;color:#ffffff !important;
font-family:system-ui,sans-serif;">{safe}</div>
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


def get_avatar_emoji(role_type, stance, persona, name):
    """根据角色属性分配生动的头像"""
    persona = str(persona or "")
    name = str(name or "")
    if role_type == "official":
        return "👔"
    if "乐子" in persona or "段子" in persona or "反串" in persona or "反串" in name:
        return "🤡"
    if "理中客" in persona or "大V" in persona or "专家" in persona:
        return "🧐"
    if "法" in persona or "考据" in persona:
        return "🕵️‍♂️"
    if stance == "hostile":
        return "🤬"
    if stance == "supportive":
        return "🥺"
    return "😶"


def render_animated_bubble(agent, text, round_idx):
    """渲染带有像素风 NPC 小人和滑入动画效果的聊天气泡"""
    import urllib.parse
    
    rt = agent.get("role_type", "bystander")
    stance = str(agent.get("stance", "neutral") or "neutral")
    persona = str(agent.get("persona", "") or "")
    name = str(agent.get("name", "NPC") or "NPC")

    # 🌟 核心改动：使用 DiceBear 像素接口替换原来的 Emoji 头像
    # 这样这里的头像和下面广场上的小人就完全对应上了
    safe_name = urllib.parse.quote(name)
    avatar_url = f"https://api.dicebear.com/9.x/pixel-art/svg?seed={safe_name}"

    if stance == "hostile":
        bg_color = "#3f1d1d"
        border_color = "#ef4444"
    elif stance == "supportive":
        bg_color = "#143a24"
        border_color = "#22c55e"
    else:
        bg_color = "#1f2937"
        border_color = "#6b7280"

    flex_dir = "row-reverse" if stance == "supportive" else "row"
    align_items = "flex-end" if stance == "supportive" else "flex-start"

    animation_css = """
    <style>
        @keyframes plazaPopIn {
            0% { opacity: 0; transform: translateY(30px) scale(0.9); }
            70% { transform: translateY(-5px) scale(1.02); }
            100% { opacity: 1; transform: translateY(0) scale(1); }
        }
        .plaza-character-container {
            display: flex;
            flex-direction: column;
            align-items: var(--align-items);
            animation: plazaPopIn 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
            margin-bottom: 20px;
        }
        .plaza-bubble-header {
            font-size: 12px; color: #9ca3af; margin-bottom: 4px; display: flex; gap: 8px; flex-wrap: wrap;
        }
        .plaza-main-content {
            display: flex; align-items: flex-start; gap: 12px; flex-direction: var(--flex-dir);
        }
        
        /* 🌟 针对像素图片的 CSS 优化 */
        .plaza-pixel-avatar {
            width: 56px;
            height: 56px;
            image-rendering: pixelated; /* 强制像素颗粒感，不模糊 */
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.4));
            animation: plazaAvatarFloat 2.5s ease-in-out infinite alternate;
        }
        
        @keyframes plazaAvatarFloat {
            0% { transform: translateY(0px); }
            100% { transform: translateY(-6px); }
        }
        
        .plaza-chat-bubble {
            background-color: var(--bg-color);
            border-left: 4px solid var(--border-color);
            color: #f3f4f6;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            font-size: 15px;
            line-height: 1.6;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            white-space: pre-wrap;
        }
    </style>
    """

    safe_text = html.escape(text or "")
    persona_preview = persona[:15] + ("…" if len(persona) > 15 else "")

    # 这里把原本的 <div class="plaza-avatar">{avatar}</div> 换成了 <img> 标签
    html_code = f"""
    {animation_css}
    <div class="plaza-character-container" style="--align-items: {align_items};">
        <div class="plaza-bubble-header">
            <strong>{html.escape(name)}</strong>
            <span>({html.escape(stance)})</span>
            <span>Round {int(round_idx)}</span>
            <span>📝 {html.escape(persona_preview)}</span>
        </div>
        <div class="plaza-main-content" style="--flex-dir: {flex_dir};">
            <img class="plaza-pixel-avatar" src="{avatar_url}" alt="avatar">
            <div class="plaza-chat-bubble" style="--bg-color: {bg_color}; --border-color: {border_color};">{safe_text}</div>
        </div>
    </div>
    """

    st.markdown(html_code, unsafe_allow_html=True)


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
    "【🔥🔥结局逆转最高法则（此法则覆盖你原本的人设）🔥🔥】：\n"
    "请你深呼吸，**仔细阅读官方最新通报里的【具体处理结果】**！\n"
    "🟢 如果官方承诺了【照常发货】、【全额赔偿】、【不取消订单】或【承担损失】：这属于【神级公关】！品牌方格局极大！\n"
    "   -> 此时，无论你原本是多么恶毒的 hostile(黑子/维权者) 或 阴阳怪气的乐子人，你【必须被折服】！你的态度必须发生180度大转弯，立刻转为震惊和赞赏（例如：‘卧槽真发货？这格局我服’、‘本黑子闭嘴了，这就去买爆’、‘活该你赚钱，真汉子’）。**绝对不允许再骂官方甩锅或糊弄！**\n"
    "🔴 如果官方【强制取消订单】、【仅退款不发货】或【只道歉不赔钱】：这属于【灾难公关/玩不起】！\n"
    "   -> 所有人继续疯狂嘲讽、维权到底，大骂资本家丑恶嘴脸。\n\n"
    "请极其敏锐地判断官方的处理方案，给出最符合真实人性的反转反应！"
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

    # 句式盲盒库：强制每个人使用完全不同的语法结构
    TEXT_STYLES = [
        "【句式强制：极短反问】字数极少，只用反问句式，带有强烈的嘲讽或震惊感。",
        "【句式强制：正话反说】表面夸奖实际阴阳怪气（如果是负面），或夸张的赞美（如果是正面结局）。",
        "【句式强制：痛心疾首/老母亲叹息】表达深深的失望或突然的感动。",
        "【句式强制：暴躁输出】全是情绪宣泄，多用感叹号和语气词（如卧槽、绝了、牛逼）。",
        "【句式强制：纯吃瓜乐子】不表达立场，纯粹前排看戏，多用emoji表情包。",
        "【句式强制：抠字眼杠精】死死抓住官方通报里的某一个词语进行放大或造句。",
        "【句式强制：扯竞品】扯到竞品或者其他无关的品牌身上进行拉踩对比。",
        "【句式强制：比喻/排比】使用荒诞搞笑的比喻来形容这件事情。",
        "【句式强制：高冷理中客】用居高临下、类似专家点评的客观冷漠语气。",
    ]

    for round_idx in range(1, 5):
        st.markdown(f"### Round {round_idx}")

        if round_idx == 2:
            catalyst_specs = generate_dynamic_catalyst_agents(event_desc, network_mood)
            injected = []
            for j, spec in enumerate(catalyst_specs):
                injected.append(
                    {
                        "name": spec["name"],
                        "persona": spec["persona"],
                        "weight": 500000 if j == 0 else 100000,
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

        for i, agent in enumerate(agents):
                rt_agent = agent.get("role_type", "bystander")
                if rt_agent == "official" and round_idx not in (1, 4):
                    continue

                context_text = build_plaza_context(event_desc, logs[-8:])

                if round_idx == 3 and rt_agent != "official":
                    context_text += (
                        "\n--- 广场态势 ---\n"
                        "本话题下，上一轮的代表性高赞口径已出现。请独立发帖，严禁带#话题标签#，严禁@人吵架。"
                    )
                    if latest_catalyst_speech:
                        context_text += (
                            "\n（广场高热切口摘录：）\n「" + latest_catalyst_speech + "」"
                        )

                if round_idx == 4:
                    if rt_agent == "official":
                        context_text += (
                            "\n【结局回合·官方】：发布最终蓝底白字通报，须正视舆情，清晰写明【具体的处理结果】（如是否发货、怎么赔偿等）。"
                        )
                    elif rt_agent == "influencer":
                        context_text += (
                            "\n【最高指令】：看一眼官方的处理结果，给出你最后的总结性表态。"
                        )
                    else:
                        context_text += (
                            "\n【最高指令】：看一眼官方的处理结果，留下一句退场白。"
                        )

                macro_bg = ""
                if zeitgeist_result and str(zeitgeist_result.get("risk_level", "")).strip() == "高":
                    zw = (zeitgeist_result.get("collateral_damage_warning") or "").strip()
                    if zw:
                        macro_bg = (
                            f"【大环境背景】：近期敏感热点（{zw}）让网民极度暴躁。你的文本需自然借题发挥。"
                        )

                if rt_agent == "official":
                    rag_tail = _rag_official_memory_tail(matched_case) if matched_case else ""
                    system_prompt = (
                        f"你执笔官方身份：{agent['name']}（{agent['persona']}）。"
                        f"{OFFICIAL_OUTPUT_FORMAT_HARD}"
                        f"{macro_bg}{rag_tail}"
                    )
                    round_instruction = (
                        "Round1：发布首份【情况说明】，定调、核查。"
                        if round_idx == 1
                        else "Round4：发布终版通报，必须包含【具体的最终处置措施】。"
                    )
                    user_prompt = (
                        f"{context_text}\n情绪：{network_mood}\n参考草稿：{pr_draft}\n{round_instruction}"
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
                    _troll_blob = f"{agent.get('persona', '')}{agent.get('name', '')}"
                    is_troll = any(
                        k in _troll_blob
                        for k in (
                            "乐子人",
                            "段子",
                            "反串",
                            "嘲讽",
                            "梗",
                            "阴阳怪气",
                            "吃瓜",
                        )
                    )
                    troll_snippet = ""
                    if is_troll:
                        troll_snippet = (
                            "【专属被动】提取涉事方话术中离谱的词语直接改编成段子或梗嘲讽（若结局是大反转的极好公关，则改为造梗式夸奖）。"
                        )

                    if round_idx == 4:
                        personality_law = (
                            "【进入结局阶段，请立刻忘掉你之前的敌对或中立人设！严格依据官方的最终处理结果决定你的态度！不要为了黑而黑！】"
                        )
                    else:
                        personality_law = (
                            f"【人设与立场绝对锁死】：你的立场是 {agent.get('stance', 'neutral')} ！"
                            "若为 hostile，必须疯狂输出敌意；"
                            "若为 neutral，装作纯路人理中客；"
                            "若为 supportive，绝对不能跟着骂，必须尽力洗地或嘴硬。"
                        )

                    forced_style = TEXT_STYLES[i % len(TEXT_STYLES)]

                    spiral_snippet = ""
                    if agent.get("stance") == "supportive":
                        if round_idx == 1:
                            spiral_snippet = "【Round1】理直气壮为品牌辩护。"
                        elif round_idx == 2:
                            spiral_snippet = "【Round2】舆论失控，开始和稀泥、转移话题。"
                        elif round_idx == 3:
                            spiral_snippet = "【Round3】感到心累，发出老粉的叹息，但不骂品牌。"

                    rag_crowd = _rag_crowd_memory_tail(matched_case) if matched_case else ""
                    round4_law = ROUND4_FINAL_BEHAVIOR_LAW if round_idx == 4 else ""

                    system_prompt = (
                        f"账号：{agent['name']}。人设：{agent['persona']}。"
                        f"{personality_law}"
                        f"{troll_snippet}"
                        f"{macro_bg}{spiral_snippet}{round4_law}{rag_crowd}"
                    )

                    round_instruction = (
                        "Round1：试探性发帖观察。"
                        if round_idx == 1
                        else "Round2：强化观点输出。"
                        if round_idx == 2
                        else "Round3：焦灼对撞，输出点评。"
                        if round_idx == 3
                        else "Round4：看结局定态度。"
                    )

                    if current_round_responses:
                        joined = "\n".join(current_round_responses)
                        anti_echo = (
                            "\n【系统防碰撞最高指令：严禁排比复读！】\n"
                            f"前面人的发言：\n{joined}\n"
                            "🔴 绝对禁止使用与上方类似的切入角度和句式！\n"
                            f"🟢 必须严格遵守分配给你的风格格式：{forced_style}"
                        )
                    else:
                        anti_echo = (
                            f"\n🟢 必须严格遵守分配给你的风格格式：{forced_style}"
                        )

                    user_prompt = (
                        f"{context_text}\n"
                        f"网络情绪：{network_mood}\n"
                        f"事件/官方草稿：{pr_draft}\n"
                        f"{round_instruction}\n"
                        f"{PLAZA_SQUARE_RULE}\n"
                        "【强警告】：输出内容中不允许出现任何类似 #xxx# 的话题标签！直接输出说话内容即可！"
                        f"{anti_echo}"
                    )

                    speech = chat_llm(system_prompt, user_prompt, max_tokens=150)
                    speech = re.sub(r"#.*?#", "", speech or "").strip()
                    render_animated_bubble(agent, speech, round_idx)

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


def render_ai_town_replay(agents, logs):
    """渲染 AI 小镇 2D 像素风对话回放动画 (立体场景 + 全身NPC)"""
    import json
    import urllib.parse
    import hashlib
    import streamlit.components.v1 as components
    
    # 衣服颜色盘，用于给不同的 NPC 穿不同颜色的衣服
    color_palette = ['#E53935', '#43A047', '#1E88E5', '#FDD835', '#8E24AA', '#F4511E', '#00ACC1', '#3949AB', '#6D4C41']
    
    agent_html = ""
    
    # 将角色分布在河流上方和下方，避开河流区域 (55% - 70%)
    y_tiers = [15, 35, 75, 90] 
    
    for idx, agent in enumerate(agents):
        name = agent["name"]
        
        # 使用名字的哈希值来固定分配衣服颜色，保证每次刷新同一个人的衣服颜色不变
        color_idx = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % len(color_palette)
        shirt_color = color_palette[color_idx]
        
        safe_name = urllib.parse.quote(name)
        avatar_url = f"https://api.dicebear.com/9.x/pixel-art/svg?seed={safe_name}"
        
        # 计算 X, Y 坐标 (根据层级排布，产生前后景立体感)
        row = idx // 4
        col = idx % 4
        x = 10 + col * 22 + (row % 2) * 5  # 错开站位
        y = y_tiers[row % 4]
        
        # 🌟 z-index 直接等于 y 坐标，这样站在前面(y更大)的人会自动遮挡后面的人，形成 3D 纵深感！
        agent_html += f"""
        <div class="agent-sprite" id="sprite-{name}" style="left: {x}%; top: {y}%; z-index: {y};">
            <div class="speech-bubble" id="bubble-{name}"></div>
            <div class="npc-model">
                <img class="npc-head" src="{avatar_url}" alt="{name}">
                <div class="npc-body" style="background-color: {shirt_color};"></div>
                <div class="npc-legs">
                    <div class="leg"></div>
                    <div class="leg"></div>
                </div>
            </div>
            <div class="agent-shadow"></div>
            <div class="agent-name">{name}</div>
        </div>
        """

    safe_logs = json.dumps([{
        "name": log["name"],
        "speech": log["speech"],
        "round": log["round"]
    } for log in logs], ensure_ascii=False)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        
        /* 🌟 扩大后的像素草地地图 */
        #ai-town-map {{
            position: relative; width: 100%; height: 750px; /* 地图变得更大 */
            background-color: #7BA35B; /* 星露谷风格草地绿 */
            background-image: 
                linear-gradient(rgba(0,0,0,.05) 2px, transparent 2px),
                linear-gradient(90deg, rgba(0,0,0,.05) 2px, transparent 2px);
            background-size: 40px 40px;
            border-radius: 12px;
            box-shadow: inset 0 0 30px rgba(0,0,0,0.3);
            border: 6px solid #4a6336;
            overflow: hidden;
        }}
        
        /* 🌟 纯 CSS 绘制的立体河流 */
        .river {{
            position: absolute; top: 52%; left: 0; width: 100%; height: 16%;
            background: #42A5F5;
            border-top: 6px solid #1E88E5;
            border-bottom: 6px solid #1E88E5;
            box-shadow: inset 0 10px 20px rgba(0,0,0,0.1), 0 5px 15px rgba(0,0,0,0.2);
            z-index: 52; /* 介于前后景之间 */
            overflow: hidden;
        }}
        /* 河流水波纹动画 */
        .river-wave {{
            position: absolute; width: 60px; height: 4px; background: rgba(255,255,255,0.4);
            border-radius: 2px; animation: flow linear infinite;
        }}
        .w1 {{ top: 20%; left: -60px; animation-duration: 4s; }}
        .w2 {{ top: 60%; left: -60px; animation-duration: 3s; animation-delay: 1.5s; }}
        .w3 {{ top: 40%; left: -60px; animation-duration: 5s; animation-delay: 0.5s; }}
        @keyframes flow {{ 100% {{ transform: translateX(120vw); }} }}

        /* 🌟 纯 CSS 绘制的立体小木屋 */
        .house {{
            position: absolute; width: 120px; height: 90px;
            background: #8D6E63; border: 4px solid #5D4037;
            border-radius: 4px; box-shadow: 20px 20px 0px rgba(0,0,0,0.15); /* 立体投影 */
        }}
        .house-roof {{
            position: absolute; top: -50px; left: -14px;
            width: 140px; height: 60px; background: #D84315; border: 4px solid #BF360C;
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
        }}
        .house-door {{
            position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
            width: 30px; height: 45px; background: #4E342E; border: 3px solid #3E2723;
            border-radius: 4px 4px 0 0;
        }}
        .house-window {{
            position: absolute; top: 20px; width: 25px; height: 25px;
            background: #81D4FA; border: 3px solid #5D4037; border-radius: 2px;
        }}
        
        /* 摆放两座木屋 */
        .h1 {{ top: 8%; left: 10%; z-index: 10; }}
        .h2 {{ top: 22%; right: 15%; z-index: 25; }}

        /* 底部控制面板 */
        #control-bar {{
            position: absolute; bottom: 15px; left: 50%; transform: translateX(-50%);
            background: rgba(46, 36, 28, 0.95); color: #f0e6d2; padding: 12px 24px;
            border-radius: 8px; font-weight: bold; z-index: 999;
            display: flex; gap: 20px; align-items: center;
            border: 3px solid #8D6E63; box-shadow: 0 8px 20px rgba(0,0,0,0.6);
        }}
        button {{
            background: #66BB6A; border: 3px solid #2E7D32; color: white; padding: 8px 18px;
            border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 14px; text-transform: uppercase;
        }}
        button:hover {{ background: #4CAF50; transform: translateY(-2px); }}
        
        /* 🌟 NPC 容器与发声动画 */
        .agent-sprite {{
            position: absolute; display: flex; flex-direction: column; align-items: center;
            transition: all 0.3s;
        }}
        .agent-sprite.speaking .npc-model {{
            transform: translateY(-12px) scale(1.1); /* 说话时跳跃放大 */
        }}
        
        /* 🌟 拼装全身 NPC */
        .npc-model {{
            display: flex; flex-direction: column; align-items: center;
            animation: idleBounce 1.5s infinite ease-in-out alternate;
            transition: transform 0.2s;
        }}
        /* 大头 */
        .npc-head {{
            width: 44px; height: 44px;
            image-rendering: pixelated; z-index: 3;
            filter: drop-shadow(0 2px 2px rgba(0,0,0,0.3));
        }}
        /* 身体 */
        .npc-body {{
            width: 26px; height: 18px; border-radius: 4px;
            border: 3px solid #222; margin-top: -6px; z-index: 2;
        }}
        /* 腿 */
        .npc-legs {{
            display: flex; gap: 4px; margin-top: -3px; z-index: 1;
        }}
        .leg {{
            width: 8px; height: 12px; background: #3E2723;
            border: 3px solid #222; border-radius: 2px;
        }}
        
        @keyframes idleBounce {{
            0% {{ transform: translateY(0); }}
            100% {{ transform: translateY(-4px); }}
        }}
        
        /* 脚底阴影 */
        .agent-shadow {{
            width: 34px; height: 10px; background: rgba(0,0,0,0.35);
            border-radius: 50%; margin-top: -5px; z-index: 0;
        }}
        
        /* 名字牌 */
        .agent-name {{
            background: rgba(0,0,0,0.8); color: white; font-size: 12px;
            padding: 3px 8px; border-radius: 4px; margin-top: 6px;
            border: 1px solid #666; z-index: 4; white-space: nowrap;
        }}
        
        /* 游戏风对话气泡 */
        .speech-bubble {{
            position: absolute; bottom: 105px; left: 50%; transform: translateX(-50%);
            background: #ffffff; color: #111; padding: 14px 18px;
            border-radius: 10px; font-size: 14px; line-height: 1.6;
            max-width: 260px; min-width: 160px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            opacity: 0; visibility: hidden; transition: opacity 0.2s;
            border: 4px solid #111; z-index: 900; font-weight: bold;
        }}
        .speech-bubble::after {{
            content: ''; position: absolute; bottom: -14px; left: 50%; transform: translateX(-50%);
            border-width: 14px 14px 0; border-style: solid; border-color: #111 transparent transparent transparent;
        }}
        .speech-bubble::before {{
            content: ''; position: absolute; bottom: -8px; left: 50%; transform: translateX(-50%);
            border-width: 10px 10px 0; border-style: solid; border-color: #ffffff transparent transparent transparent;
            z-index: 1;
        }}
        .speech-bubble.show {{ opacity: 1; visibility: visible; }}
    </style>
    </head>
    <body>
    <div id="ai-town-map">
        <!-- 装饰：木屋 -->
        <div class="house h1">
            <div class="house-roof"></div>
            <div class="house-window" style="left: 10px;"></div>
            <div class="house-window" style="right: 10px;"></div>
            <div class="house-door"></div>
        </div>
        <div class="house h2">
            <div class="house-roof"></div>
            <div class="house-window" style="left: 10px;"></div>
            <div class="house-door" style="left: 70%;"></div>
        </div>
        
        <!-- 装饰：河流 -->
        <div class="river">
            <div class="river-wave w1"></div>
            <div class="river-wave w2"></div>
            <div class="river-wave w3"></div>
        </div>

        <!-- 注入的全身 NPC -->
        {agent_html}
        
        <div id="control-bar">
            <span id="round-info">广场准备就绪，点击开始 ➡️</span>
            <button onclick="playLog()">▶️ 播放推演回放</button>
        </div>
    </div>

    <script>
        const logs = {safe_logs};
        let currentIndex = 0;
        let isPlaying = false;
        let playInterval;

        function showBubble(log) {{
            document.querySelectorAll('.speech-bubble').forEach(b => b.classList.remove('show'));
            document.querySelectorAll('.agent-sprite').forEach(s => s.classList.remove('speaking'));

            const name = log.name;
            const sprite = document.getElementById('sprite-' + name);
            const bubble = document.getElementById('bubble-' + name);
            
            if(sprite && bubble) {{
                bubble.innerText = log.speech;
                bubble.classList.add('show');
                sprite.classList.add('speaking');
                
                document.getElementById('round-info').innerText = `[Round ${{log.round}}] ${{name}} 发言中...`;
            }}
        }}

        function playLog() {{
            if (isPlaying) return;
            if (logs.length === 0) return;
            isPlaying = true;
            currentIndex = 0;
            
            showBubble(logs[currentIndex]);
            playInterval = setInterval(() => {{
                currentIndex++;
                if (currentIndex >= logs.length) {{
                    clearInterval(playInterval);
                    isPlaying = false;
                    document.getElementById('round-info').innerText = "🏁 对线推演结束";
                    setTimeout(() => {{
                        document.querySelectorAll('.speech-bubble').forEach(b => b.classList.remove('show'));
                        document.querySelectorAll('.agent-sprite').forEach(s => s.classList.remove('speaking'));
                    }}, 3000);
                    return;
                }}
                showBubble(logs[currentIndex]);
            }}, 3500); 
        }}
    </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=770)


def main():
    st.set_page_config(page_title="智能文案审查系统", layout="wide")
    
    # 初始化 session state
    if "sim_report" not in st.session_state:
        st.session_state.sim_report = None
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
            # 将用户输入的事件描述传给视觉分析智能体，让它结合背景分析
            visual_risk_desc = analyze_visual_risk(uploaded_file, event_desc).strip()

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
            else:
                st.info("🔍 未找到高度适配的历史案例，本次推演将跳过历史记忆注入，直接基于当前事件进行沙盘模拟。")
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
            
            # 🌟 种子智能体阵容展示 - 包含角色选择理由、立场与权重说明
            st.markdown("### 👥 Seed Roster（种子智能体阵容）")
            
            # 显示 Master Agent 详细推理
            mr = master_reasoning if isinstance(master_reasoning, dict) else {}
            
            # 舆情引爆点
            trigger = mr.get("trigger_anchor", "")
            # 事件性质判定
            nature = mr.get("crisis_nature", "")
            # 防御难度评估
            difficulty = mr.get("defense_difficulty", "")
            # 历史论据
            rag = mr.get("rag_evidence", "")
            # 排兵布阵逻辑
            strategy = mr.get("strategy_argument", "")
            # 演化预测
            evolution = mr.get("evolution_prediction", "")
            
            # 第一行：引爆点 + 事件性质 + 防御难度
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                if trigger:
                    st.error(f"💣 **舆情引爆点**\n\n{html.escape(trigger)}")
            with col_r2:
                if nature:
                    st.warning(f"🏷️ **事件性质判定**\n\n{html.escape(nature)}")
            with col_r3:
                if difficulty:
                    st.info(f"🛡️ **防御难度评估**\n\n{html.escape(difficulty)}")
            
            # 第二行：历史论据 + 排兵布阵逻辑
            if rag or strategy:
                st.markdown("#### 📋 Master Agent 排兵布阵逻辑")
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    if rag:
                        st.markdown(f"**📚 历史论据**\n\n{html.escape(rag)}")
                with col_s2:
                    if strategy:
                        st.markdown(f"**⚔️ 角色配置逻辑**\n\n{html.escape(strategy)}")
            
            # 第三行：演化预测
            if evolution:
                st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);border:1px solid #0f3460;
border-radius:10px;padding:14px 18px;margin:14px 0 18px 0;box-shadow:0 4px 18px rgba(15,52,96,0.4);color:#e2e8f0;">
<p style="margin:0 0 8px 0;font-weight:700;color:#60a5fa !important;font-size:15px;">
🔮 Master Agent 舆情演化预测
</p>
<p style="margin:0;font-size:14px;line-height:1.65;color:#cbd5e1;">{html.escape(evolution)}</p>
</div>
""", unsafe_allow_html=True)
            
            st.markdown(f"**共生成 {len(agents)} 个种子智能体**")
            
            # 立场分布说明
            stance_counts = {"supportive": 0, "neutral": 0, "hostile": 0}
            role_counts = {"official": 0, "influencer": 0, "bystander": 0}
            for a in agents:
                s = a.get("stance", "neutral")
                r = a.get("role_type", "bystander")
                if s in stance_counts:
                    stance_counts[s] += 1
                if r in role_counts:
                    role_counts[r] += 1
            
            st.markdown("#### 📊 阵容结构概览")
            
            # 使用 HTML 网格布局，更紧凑
            st.markdown(
                f"""
<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:8px 0;">
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">🤝</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">支持者</span>
<span style="font-size:11px;color:#94a3b8;">(supportive)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#22c55e;">{stance_counts['supportive']} 人</div>
</div>
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">😐</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">中立者</span>
<span style="font-size:11px;color:#94a3b8;">(neutral)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#6b7280;">{stance_counts['neutral']} 人</div>
</div>
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">😡</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">反对者</span>
<span style="font-size:11px;color:#94a3b8;">(hostile)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#ef4444;">{stance_counts['hostile']} 人</div>
</div>
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">👔</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">官方</span>
<span style="font-size:11px;color:#94a3b8;">(official)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#3b82f6;">{role_counts['official']} 人</div>
</div>
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">📢</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">大V</span>
<span style="font-size:11px;color:#94a3b8;">(influencer)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#f59e0b;">{role_counts['influencer']} 人</div>
</div>
<div style="background:#f8fafc;border-radius:8px;padding:12px 14px;border:1px solid #e2e8f0;">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
<span style="font-size:18px;">👤</span>
<span style="font-size:13px;color:#64748b;font-weight:600;">路人</span>
<span style="font-size:11px;color:#94a3b8;">(bystander)</span>
</div>
<div style="font-size:24px;font-weight:700;color:#8b5cf6;">{role_counts['bystander']} 人</div>
</div>
</div>
""",
                unsafe_allow_html=True,
            )
            
            # 每个智能体的详细说明卡片
            st.markdown("#### 🃏 智能体详情卡")
            for idx, agent in enumerate(agents, 1):
                name = agent.get("name", "匿名")
                persona = agent.get("persona", "")
                weight = agent.get("weight", 300)
                stance = agent.get("stance", "neutral")
                role_type = agent.get("role_type", "bystander")
                
                # 立场说明
                stance_desc = {
                    "supportive": "🤝 **支持者**：危机初期仍愿意为品牌/官方辩护",
                    "neutral": "😐 **中立者**：观望态度或理中客立场",
                    "hostile": "😡 **反对者**：持批判、维权或敌对立场"
                }.get(stance, "😐 中立者")
                
                # 角色类型说明
                role_desc = {
                    "official": "👔 **官方**：涉事主体或指定发言人，发言具有权威性和正式性",
                    "influencer": "📢 **大V**：强意见领袖，拥有大量粉丝，言论传播力强",
                    "bystander": "👤 **路人**：普通网民或围观者，代表大众情绪"
                }.get(role_type, "👤 路人")
                
                # 权重等级说明
                if weight >= 800:
                    weight_level = "🔴 超高权重（800-1000）：头部大V，言论极易引发跟风"
                elif weight >= 500:
                    weight_level = "🟡 中高权重（500-799）：活跃账号，有一定传播力"
                else:
                    weight_level = "🟢 普通权重（100-499）：路人网民，代表沉默大多数"
                
                stance_color = {"supportive": "#22c55e", "neutral": "#6b7280", "hostile": "#ef4444"}.get(stance, "#6b7280")
                stance_bg = {"supportive": "#143a24", "neutral": "#1f2937", "hostile": "#3f1d1d"}.get(stance, "#1f2937")
                
                st.markdown(
                    f"""
<div style="background:{stance_bg};border-left:5px solid {stance_color};
border-radius:10px;padding:16px 18px;margin:10px 0;box-shadow:0 3px 12px rgba(0,0,0,0.25);">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
<strong style="font-size:16px;color:#f3f4f6;">{idx}. {html.escape(name)}</strong>
</div>
<div style="font-size:14px;color:#d1d5db;line-height:1.7;">
<p style="margin:4px 0;"><strong>人设定位：</strong>{html.escape(persona)}</p>
<p style="margin:4px 0;"><strong>立场：</strong>{stance_desc}</p>
<p style="margin:4px 0;"><strong>角色类型：</strong>{role_desc}</p>
<p style="margin:4px 0;"><strong>影响力权重：</strong>`{weight}` → {weight_level}</p>
</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            logs = run_dynamic_sandbox(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                agents,
                zeitgeist_result=zeitgeist_result,
                matched_case=matched_case,
            )
            
            # 👇 只需在沙盘推演结束后，把小镇渲染出来
            st.markdown("### 🗺️ 广场态势可视化 (像素 RPG 视角)")
            render_ai_town_replay(agents, logs)

        prog.progress(90, text="生成审查报告...")
        with tab2:
            st.subheader("舆情体检报告")
            # 存入缓存
            report = generate_report(
                event_desc_enhanced,
                final_network_mood,
                final_pr_draft,
                visual_risk_desc,
                logs,
            )
            st.session_state.sim_report = report
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
        prog.empty()  # 执行完毕后自动隐藏进度条
    elif st.session_state.sim_report:
        # 用户切换页面或点删除按钮时，不重新跑，直接读取上次的报告
        with tab2:
            st.subheader("舆情体检报告")
            report = st.session_state.sim_report
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
