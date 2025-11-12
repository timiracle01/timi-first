###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import json
import requests
import sys
import csv
import re
import time
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 配置大模型API参数
client = OpenAI(
    api_key="dab88b90d1466275d34b5af41eab74d4aff5768d",
    base_url="https://aistudio.baidu.com/llm/lmapi/v3"
)

# 全局缓存
embedding_cache = {}
cache_lock = threading.Lock()

# 预计算知识库嵌入向量
knowledge_base_embeddings = None
def parse_response(content):
    """
    解析大模型返回的响应内容
    处理包含代码块的JSON响应
    """
    # 检查是否包含代码块
    # print(content)
    code_block_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    # print(code_block_pattern)
    match = code_block_pattern.search(content)

    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {json_str}")
            return None
    else:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"直接JSON解析失败: {content}")
            return None

def get_embedding(text):
    """
    获取文本的向量表示（带缓存）
    """
    # 检查缓存
    with cache_lock:
        if text in embedding_cache:
            return embedding_cache[text]

    # 批量获取多个文本的嵌入向量
    if isinstance(text, list):
        return get_embeddings_batch(text)

    url = "https://qianfan.baidubce.com/v2/embeddings"
    payload = json.dumps({
        "model": "bge-large-zh",
        "input": [text]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-5tfX41HMR5wReJjGJL1pP/ec6d6701e0a9b84dd4951ce977f2a9bc5c624d53'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        embedding = data['data'][0]['embedding']

        # 更新缓存
        with cache_lock:
            embedding_cache[text] = embedding

        return embedding
    except Exception as e:
        print(json.dumps({
            "code": 1,
            "errorMsg": f"Embedding API error: {str(e)}",
            "score": 0.0,
            "data": [{"score": 0}]
        }, ensure_ascii=False), flush=True)
        return None

def get_embeddings_batch(texts):
    """
    批量获取嵌入向量
    """
    # 过滤已缓存的文本
    uncached_texts = []
    embeddings = []

    with cache_lock:
        for text in texts:
            if text in embedding_cache:
                embeddings.append(embedding_cache[text])
            else:
                uncached_texts.append(text)

    # 如果没有需要获取的文本，直接返回
    if not uncached_texts:
        return embeddings

    url = "https://qianfan.baidubce.com/v2/embeddings"
    payload = json.dumps({
        "model": "bge-large-zh",
        "input": uncached_texts
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer bce-v3/ALTAK-5tfX41HMR5wReJjGJL1pP/ec6d6701e0a9b84dd4951ce977f2a9bc5c624d53'
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        # 更新缓存
        with cache_lock:
            for i, text in enumerate(uncached_texts):
                embedding = data['data'][i]['embedding']
                embedding_cache[text] = embedding
                embeddings.append(embedding)

        return embeddings
    except Exception as e:
        print(f"批量嵌入API错误: {str(e)}")
        return [None] * len(texts)

def calculate_similarity(text1, text2):
    """
    计算两个文本的相似度（优化版）
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        # 批量获取嵌入向量
        embeddings = get_embeddings_batch([text1, text2])
        if None in embeddings:
            return 0.0

        emb1, emb2 = embeddings
        sim = cosine_similarity(
            np.array(emb1).reshape(1, -1),
            np.array(emb2).reshape(1, -1)
        )
        return float(sim[0][0])
    except Exception as e:
        print(json.dumps({
            "code": 1,
            "errorMsg": f"Similarity calculation error: {str(e)}",
            "score": 0.0,
            "data": [{"score": 0}]
        }, ensure_ascii=False), flush=True)
        return 0.0

def calculate_similarity_with_embeddings(emb1, emb2):
    """
    使用预计算的嵌入向量计算相似度
    """
    if emb1 is None or emb2 is None:
        return 0.0
    try:
        sim = cosine_similarity(
            np.array(emb1).reshape(1, -1),
            np.array(emb2).reshape(1, -1)
        )
        return float(sim[0][0])
    except Exception as e:
        return 0.0
def build_knowledge_base():
    """
    构建症状-证型-治法知识库（带预计算嵌入向量）
    """
    train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
    knowledge_base = []

    with open(train_data, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                knowledge_base.append({
                    'symptoms': row[1],
                    'zx': row[2],
                    'zf': row[3]
                })

    # 预计算知识库症状的嵌入向量
    global knowledge_base_embeddings
    symptom_texts = [item['symptoms'] for item in knowledge_base]
    knowledge_base_embeddings = get_embeddings_batch(symptom_texts)

    # 将嵌入向量添加到知识库
    for i, item in enumerate(knowledge_base):
        item['embedding'] = knowledge_base_embeddings[i]

    return knowledge_base

def find_best_match(symptoms, knowledge_base):
    """
    基于相似度找到最佳匹配的证型和治法（优化版）
    """
    best_similarity = 0.0
    best_match = {'zx': '未知', 'zf': '待定'}

    # 获取症状的嵌入向量
    symptom_embedding = get_embedding(symptoms)
    if symptom_embedding is None:
        return best_match, best_similarity

    # 使用预计算的嵌入向量进行相似度计算
    for item in knowledge_base:
        similarity = calculate_similarity_with_embeddings(symptom_embedding, item.get('embedding'))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {'zx': item['zx'], 'zf': item['zf']}

    return best_match, best_similarity

def find_best_match_batch(symptoms_list, knowledge_base):
    """
    批量查找最佳匹配
    """
    # 批量获取症状嵌入向量
    symptom_embeddings = get_embeddings_batch(symptoms_list)

    results = []
    for symptom_embedding in symptom_embeddings:
        best_similarity = 0.0
        best_match = {'zx': '未知', 'zf': '待定'}

        if symptom_embedding is None:
            results.append((best_match, best_similarity))
            continue

        for item in knowledge_base:
            similarity = calculate_similarity_with_embeddings(symptom_embedding, item.get('embedding'))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {'zx': item['zx'], 'zf': item['zf']}

        results.append((best_match, best_similarity))

    return results

def call_large_model(symptoms, knowledge_base):
    """
    改进的LLM调用：结合知识库匹配和LLM预测
    返回结构化JSON结果
    """
    # 首先尝试知识库匹配
    kb_match, kb_similarity = find_best_match(symptoms, knowledge_base)

    # 如果相似度足够高，直接使用知识库结果
    if kb_similarity > 0.7:
        return {"证型": kb_match['zx'], "治法": kb_match['zf']}

    # 否则使用LLM预测，但提供更精确的提示
    system_prompt = """
    你是一位经验丰富的中医专家，专门治疗慢性淋巴细胞白血病。

    请根据患者症状描述判断证型和治法，必须严格遵循以下中医辨证标准：

    常见证型及对应治法：
    1. 痰湿内蕴 → 健脾燥湿，化痰利浊
    2. 脾虚痰湿 → 健脾益气，化湿祛痰
    3. 气阴两虚 → 益气养阴，生津润燥
    4. 痰瘀互结 → 豁痰祛瘀，软坚散结
    5. 痰湿内蕴兼气虚发热 → 健脾燥湿，化痰利浊

    判断要点：
    - 痰湿内蕴：口中发黏，痰涎量多，胸闷腹胀，舌淡红齿痕
    - 脾虚痰湿：形体肥胖，动时即乏，不欲食，舌胖大
    - 气阴两虚：口干口苦，手足心热，腰膝酸软，睡眠不安
    - 痰瘀互结：肿核质硬，活动度差，舌暗红有淤点

    请严格从上述证型中选择，治法必须与证型完全对应。
    请用JSON格式输出：{"证型":"", "治法":""}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"患者症状：{symptoms}"}
    ]

    try:
        response = client.chat.completions.create(
            model="ernie-4.5-0.3b",
            messages=messages,
            max_completion_tokens=512,
            temperature=0.1,  # 降低温度以提高一致性
            top_p=0.5
        )

        # 解析响应内容
        content = response.choices[0].message.content
        result = parse_response(content)

        if result:
            # 后处理：确保证型和治法符合训练数据模式
            result = post_process_result(result, knowledge_base)
            return result
        else:
            # 如果LLM失败，使用知识库匹配
            return {"证型": kb_match['zx'], "治法": kb_match['zf']}

    except Exception as e:
        print(f"API调用异常: {str(e)}")
        return {"证型": kb_match['zx'], "治法": kb_match['zf']}

def post_process_result(result, knowledge_base):
    """
    后处理：确保LLM输出符合训练数据模式
    """
    # 定义标准证型和治法
    standard_zx = ['痰湿内蕴', '脾虚痰湿', '气阴两虚', '痰瘀互结', '痰湿内蕴兼气虚发热']
    standard_zf = ['健脾燥湿，化痰利浊', '健脾益气，化湿祛痰', '益气养阴，生津润燥', '豁痰祛瘀，软坚散结']

    zx = result.get('证型', '')
    zf = result.get('治法', '')

    # 如果证型不在标准列表中，找到最相似的
    if zx not in standard_zx:
        best_similarity = 0.0
        best_zx = '痰湿内蕴'  # 默认值
        for standard in standard_zx:
            similarity = calculate_similarity(zx, standard)
            if similarity > best_similarity:
                best_similarity = similarity
                best_zx = standard
        result['证型'] = best_zx

    # 如果治法不在标准列表中，根据证型映射
    if zf not in standard_zf:
        zx_to_zf = {
            '痰湿内蕴': '健脾燥湿，化痰利浊',
            '脾虚痰湿': '健脾益气，化湿祛痰',
            '气阴两虚': '益气养阴，生津润燥',
            '痰瘀互结': '豁痰祛瘀，软坚散结',
            '痰湿内蕴兼气虚发热': '健脾燥湿，化痰利浊'
        }
        result['治法'] = zx_to_zf.get(result['证型'], '健脾燥湿，化痰利浊')

    return result
# 构建知识库
knowledge_base = build_knowledge_base()

# 读取训练数据用于评分
train_data = './datasets/68f201a04e0f8ad44a62069b-momodel/train_data.csv'
symptoms_data = []
ZX = []  # 证型
ZF = []  # 治法

with open(train_data, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        symptoms_data.append(row[1])  # 提取症状列
        ZX.append(row[2])  # 提取证型列
        ZF.append(row[3])  # 提取治法列

# 生成新的结果数据
predict_zx = []
predict_zf = []

def predict(symptom):
    model_result = call_large_model(symptom, knowledge_base)
    return model_result['证型'], model_result['治法']

def predict_batch(symptoms_list):
    """
    批量预测
    """
    results = []

    # 批量进行知识库匹配
    kb_matches = find_best_match_batch(symptoms_list, knowledge_base)

    for i, (symptom, (kb_match, kb_similarity)) in enumerate(zip(symptoms_list, kb_matches)):
        # 如果相似度足够高，直接使用知识库结果
        if kb_similarity > 0.7:
            results.append((kb_match['zx'], kb_match['zf']))
        else:
            # 否则使用LLM预测
            model_result = call_large_model(symptom, knowledge_base)
            results.append((model_result['证型'], model_result['治法']))

    return results

print("开始预测...")

# 批量处理，每批处理10个症状
batch_size = 10
for i in range(0, len(symptoms_data), batch_size):
    batch_symptoms = symptoms_data[i:i+batch_size]
    batch_results = predict_batch(batch_symptoms)

    for zx, zf in batch_results:
        predict_zx.append(zx)
        predict_zf.append(zf)

    if i + batch_size >= len(symptoms_data):
        print(f"已完成 {len(symptoms_data)}/{len(symptoms_data)} 条预测")
    else:
        print(f"已完成 {i+batch_size}/{len(symptoms_data)} 条预测")

print("预测完成")
def score(predict_zx, predict_zf):
    ZX_score = []
    ZF_score = []

    # 批量计算相似度
    zx_pairs = [(predict_zx[i].strip(), ZX[i].strip()) for i in range(len(predict_zx))]
    zf_pairs = [(predict_zf[i].strip(), ZF[i].strip()) for i in range(len(predict_zf))]

    # 批量计算证型相似度
    zx_texts = [pair[0] for pair in zx_pairs] + [pair[1] for pair in zx_pairs]
    zx_embeddings = get_embeddings_batch(zx_texts)

    half_len = len(zx_pairs)
    for i in range(half_len):
        emb1 = zx_embeddings[i]
        emb2 = zx_embeddings[i + half_len]
        sim_x = calculate_similarity_with_embeddings(emb1, emb2)
        ZX_score.append(sim_x)

    # 批量计算治法相似度
    zf_texts = [pair[0] for pair in zf_pairs] + [pair[1] for pair in zf_pairs]
    zf_embeddings = get_embeddings_batch(zf_texts)

    half_len = len(zf_pairs)
    for i in range(half_len):
        emb1 = zf_embeddings[i]
        emb2 = zf_embeddings[i + half_len]
        sim_f = calculate_similarity_with_embeddings(emb1, emb2)
        ZF_score.append(sim_f)

    zx_mean = np.mean(ZX_score) if ZX_score else 0.0
    zf_mean = np.mean(ZF_score) if ZF_score else 0.0

    final_score = ((zx_mean + zf_mean) / 2) * 100  # 百分制
    return final_score
final_score = score(predict_zx, predict_zf)
print(final_score)
from openai import OpenAI
#  实现大模型调用方法

#  进行推理输出相应内容
def predict(symptom):
    """
    :param symptom: str, 症状描述
    :return zx_predict, zf_predict：str, 依次为症型，治法描述
    """
    #  根据输入的症状，使用大模型推理相应的内容


    return zx_predict, zf_predict
