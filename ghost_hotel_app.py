"""
幽灵酒店检测平台
基于文本相似度和地理距离检测潜在的违规房源
"""

import streamlit as st
import pandas as pd
import numpy as np
from geopy.distance import geodesic

from gensim import corpora, models, similarities
import requests
import pydeck as pdk
import os

# 页面配置
st.set_page_config(
    page_title="幽灵酒店检测平台",
    page_icon="👻",
    layout="wide"
)

# ===================== 停用词 =====================

@st.cache_resource
def load_stopwords():
    """
    加载停用词：
    - 内置一份常见英文停用词表（不依赖 NLTK）
    - 再叠加伦敦空间停用词表
    """
    # 内置英文停用词（可以后续再扩展）
    ENGLISH_STOPWORDS = {
        "a", "an", "the", "and", "or", "but",
        "of", "to", "in", "on", "for", "with",
        "at", "by", "from", "up", "about", "into",
        "over", "after", "before", "between", "out",
        "during", "without", "through", "above", "below",
        "is", "am", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did",
        "this", "that", "these", "those", "it", "its",
        "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our",
        "as", "so", "than", "too", "very",
        "can", "could", "should", "would",
        "will", "just", "not", "no", "nor",
        "there", "here", "when", "where", "why", "how",
        # 常见 Airbnb 描述里的弱信息词，可以慢慢加
        "room", "rooms", "flat", "apartment", "house",
        "studio", "bedroom", "bathroom",
        "london", "central", "city", "center", "centre"
    }

    sw = set(ENGLISH_STOPWORDS)

    # 叠加伦敦空间停用词
    try:
        response = requests.get(
            'https://raw.githubusercontent.com/YUJIA-MA-UCL/LondonGhostHotel_DetectionAPP/refs/heads/main/data/London_Spatial_Stopwords_list_1126.CSV',
            timeout=10
        )
        response.raise_for_status()
        stopWords_london = set(response.text.split())
        sw = sw | stopWords_london
    except Exception as e:
        st.warning(f"无法加载伦敦停用词列表: {e}")

    return sw


stopWords = load_stopwords()

# ===================== 数据加载 =====================

@st.cache_data
def load_ghost_hotel_data():
    """加载全部 Airbnb listing 数据（作为对照基准）"""
    data_paths = [
        'data/df.csv',
        'data/df.csv.gz',
        'https://raw.githubusercontent.com/YUJIA-MA-UCL/LondonGhostHotel_DetectionAPP/refs/heads/main/data/df.csv.gz'
    ]

    for path in data_paths:
        try:
            if path.startswith("http"):
                if path.endswith(".gz"):
                    df = pd.read_csv(path, compression="gzip")
                else:
                    df = pd.read_csv(path)
            else:
                if not os.path.exists(path):
                    continue
                if path.endswith(".gz"):
                    df = pd.read_csv(path, compression="gzip")
                else:
                    df = pd.read_csv(path)

            required = ['id', 'description', 'neighborhood_overview', 'latitude', 'longitude']
            if all(col in df.columns for col in required):
                return df
        except Exception:
            continue

    return None

@st.cache_resource
def build_text_indices(listings_df: pd.DataFrame):
    """
    基于全体 listings_df 预先构建两套 TF-IDF 索引：
    - description 通道
    - neighborhood_overview 通道

    返回一个元组：
    (desc_dict, desc_tfidf, desc_index,
     ov_dict,   ov_tfidf,   ov_index)
    若某一通道完全没有有效文本，则对应位置为 None。
    """
    if listings_df is None or listings_df.empty:
        return None

    # -------- description 通道 --------
    if "description" in listings_df.columns:
        desc_texts = listings_df["description"].fillna("").astype(str).tolist()
        desc_tokens_list = [
            [w for w in simple_tokenize(t) if w not in stopWords]
            for t in desc_texts
        ]
        non_empty_desc = [ws for ws in desc_tokens_list if ws]
        if len(non_empty_desc) > 0:
            desc_dict = corpora.Dictionary(non_empty_desc)
            desc_corpus = [desc_dict.doc2bow(ws) for ws in desc_tokens_list]
            desc_tfidf = models.TfidfModel(desc_corpus)
            desc_index = similarities.MatrixSimilarity(
                desc_tfidf[desc_corpus],
                num_features=len(desc_dict),
            )
        else:
            desc_dict = desc_tfidf = desc_index = None
    else:
        desc_dict = desc_tfidf = desc_index = None

    # -------- neighborhood_overview 通道 --------
    if "neighborhood_overview" in listings_df.columns:
        ov_texts = listings_df["neighborhood_overview"].fillna("").astype(str).tolist()
        ov_tokens_list = [
            [w for w in simple_tokenize(t) if w not in stopWords]
            for t in ov_texts
        ]
        non_empty_ov = [ws for ws in ov_tokens_list if ws]
        if len(non_empty_ov) > 0:
            ov_dict = corpora.Dictionary(non_empty_ov)
            ov_corpus = [ov_dict.doc2bow(ws) for ws in ov_tokens_list]
            ov_tfidf = models.TfidfModel(ov_corpus)
            ov_index = similarities.MatrixSimilarity(
                ov_tfidf[ov_corpus],
                num_features=len(ov_dict),
            )
        else:
            ov_dict = ov_tfidf = ov_index = None
    else:
        ov_dict = ov_tfidf = ov_index = None

    return (desc_dict, desc_tfidf, desc_index,
            ov_dict,   ov_tfidf,   ov_index)

# ===================== 文本相似度相关函数 =====================
import re

def simple_tokenize(text):
    """
    - 只保留字母数字
    - 全部小写
    - 不会报 punkt / punkt_tab 错误
    """
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def gensimilarities(test, textList):
    """
    使用 TF-IDF 计算 test 与 textList 中各文本的相似度
    """
    textList = ["" if t is None else str(t) for t in textList]
    N = len(textList)

    # 文本全部分词（不丢失原始顺序）
    allWordsList = []
    for text in textList:
        tokens = [w for w in simple_tokenize(text) if w not in stopWords]
        allWordsList.append(tokens)

    # 如果所有文本都没词，直接返回全 0
    if all(len(ws) == 0 for ws in allWordsList):
        return np.zeros(N, float)

    # 构造字典（跳过空 tokens）
    non_empty = [ws for ws in allWordsList if ws]
    dictionary = corpora.Dictionary(non_empty)
    if len(dictionary) == 0:
        return np.zeros(N, float)

    # 各文本的 BOW
    corpus = [dictionary.doc2bow(ws) for ws in allWordsList]

    # test 文本 → tokens
    test_tokens = [w for w in simple_tokenize(test) if w not in stopWords]
    if not test_tokens:
        return np.zeros(N, float)

    test_bow = dictionary.doc2bow(test_tokens)

    # TF-IDF & similarity
    tfidf = models.TfidfModel(corpus)
    index = similarities.MatrixSimilarity(tfidf[corpus], num_features=len(dictionary))
    sim = index[tfidf[test_bow]]

    return np.array(sim, float)


def best_similarity_with_candidates(
    input_description: str,
    input_overview: str,
    candidate_descriptions: list,
    candidate_overviews: list
):
    """
    用“输入的描述 / overview”对一组候选文本做相似度比对，
    返回：
    - best_sim: 所有候选中的最高相似度
    - best_idx: 对应的候选索引（在 candidate_* 列表里的位置）
    - sims_desc: 按候选顺序的 description 相似度数组（numpy.array 或 None）
    """
    best_sim = 0.0
    best_idx = None

    sims_desc = None
    sims_ov = None

    # description 通道
    if input_description and input_description.strip() and candidate_descriptions:
        sims_desc = gensimilarities(input_description, candidate_descriptions)

    # overview 通道
    if input_overview and input_overview.strip() and candidate_overviews:
        sims_ov = gensimilarities(input_overview, candidate_overviews)

    max_desc = float(np.max(sims_desc)) if sims_desc is not None and len(sims_desc) > 0 else 0.0
    idx_desc = int(np.argmax(sims_desc)) if sims_desc is not None and len(sims_desc) > 0 else None

    max_ov = float(np.max(sims_ov)) if sims_ov is not None and len(sims_ov) > 0 else 0.0
    idx_ov = int(np.argmax(sims_ov)) if sims_ov is not None and len(sims_ov) > 0 else None

    if max_ov >= max_desc:
        best_sim = max_ov
        best_idx = idx_ov
    else:
        best_sim = max_desc
        best_idx = idx_desc

    return best_sim, best_idx, sims_desc


# ===================== 空间距离相关函数 =====================

def check_geographic_proximity(lat, lon, listings_df, threshold_meters=200):
    """
    检查地理位置是否接近任意已有 Airbnb listing
    返回 threshold_meters 内的 listing 列表
    """
    if listings_df is None or listings_df.empty:
        return []

    nearby_listings = []
    test_location = (lat, lon)

    for idx, row in listings_df.iterrows():
        try:
            ll_lat = float(row['latitude'])
            ll_lon = float(row['longitude'])
            ll_location = (ll_lat, ll_lon)

            distance = geodesic(test_location, ll_location).meters

            if distance <= threshold_meters:
                nearby_listings.append({
                    'id': row.get('id', 'N/A'),
                    'host_id': row.get('host_id', 'N/A'),
                    'distance_meters': round(distance, 2),
                    'description': row.get('description', 'N/A'),
                    'neighborhood_overview': row.get('neighborhood_overview', 'N/A')
                })
        except (ValueError, KeyError):
            continue

    return nearby_listings


# ===================== 核心检测逻辑 =====================
def detect_ghost_hotel(
    host_id: str,
    description: str,
    neighborhood_overview: str,
    latitude: float,
    longitude: float,
    listings_df: pd.DataFrame,
    distance_threshold: float = 200,
    similarity_threshold: float = 0.5,
):
    """
    按 “同一 host_id + 距离 + 文本相似度” 检测潜在幽灵酒店。

    - 先在 listings_df 中筛选出 host_id 相同的所有房源
    - 在这些房源中，找出与输入位置距离 <= distance_threshold 的房源
    - 对这些“同 host 且近距离”的房源做文本相似度；相似度 >= similarity_threshold 的视为高风险
    """

    results = {
        "is_potential_ghost_hotel": False,
        "geographic_match": False,
        "description_match": False,
        "similarity_score": 0.0,
        "nearby_listings": [],
        "similar_listings": [],
        "best_match": None,
        "all_similarities": [],
        "matched_count": 0,
        "details": {},
    }

    # 0. 基础数据校验
    if listings_df is None or listings_df.empty:
        return results

    # 1️⃣ 先筛出同一个 host_id 的房源
    if "host_id" not in listings_df.columns:
        results["details"]["error"] = "listings_df 中没有 host_id 列"
        return results

    host_str = str(host_id)
    same_host_df = listings_df[listings_df["host_id"].astype(str) == host_str].copy()

    if same_host_df.empty:
        # 没有找到同 host 的其他房源，直接返回
        return results

    # 2️⃣ 在同一个 host 的房源中做距离筛选
    nearby = check_geographic_proximity(
        latitude,
        longitude,
        same_host_df,
        threshold_meters=distance_threshold,
    )
    results["nearby_listings"] = nearby
    results["geographic_match"] = len(nearby) > 0

    # 如果同 host 内根本没有 200m 内的其他房源，那按你的新逻辑可以直接结束
    if not nearby:
        results["is_potential_ghost_hotel"] = False
        return results

    # 3️⃣ 从 same_host_df 中构建候选文本（全部同 host）
    cand_desc = same_host_df["description"].fillna("").astype(str).tolist() \
        if "description" in same_host_df.columns else []
    cand_ov = same_host_df["neighborhood_overview"].fillna("").astype(str).tolist() \
        if "neighborhood_overview" in same_host_df.columns else []

    if not cand_desc and not cand_ov:
        # 没有文本可比，只有地理上的“可疑”
        results["is_potential_ghost_hotel"] = results["geographic_match"]
        return results

    # 4️⃣ 文本相似度：输入 vs 同 host 所有房源
    sims_desc = gensimilarities(description, cand_desc) if cand_desc else np.zeros(len(same_host_df))
    sims_desc = np.array(sims_desc, dtype=float)

    if neighborhood_overview and neighborhood_overview.strip() and cand_ov:
        sims_ov = gensimilarities(neighborhood_overview, cand_ov)
        sims_ov = np.array(sims_ov, dtype=float)
    else:
        sims_ov = np.zeros(len(same_host_df), dtype=float)

    # 每一条 listing 的最终相似度：desc 和 overview 的逐条 max
    all_similarities = np.maximum(sims_desc, sims_ov)
    results["all_similarities"] = all_similarities.tolist()

    # 5️⃣ 找到最高相似度及对应房源
    max_sim = float(all_similarities.max()) if len(all_similarities) > 0 else 0.0
    results["similarity_score"] = max_sim

    if len(all_similarities) > 0:
        best_idx = int(all_similarities.argmax())
        best_row = same_host_df.iloc[best_idx]

        # 计算 best_match 离输入房源的距离
        try:
            gh_lat = float(best_row.get("latitude", np.nan))
            gh_lon = float(best_row.get("longitude", np.nan))
            if not np.isnan(gh_lat) and not np.isnan(gh_lon):
                dist_best = geodesic((latitude, longitude), (gh_lat, gh_lon)).meters
            else:
                dist_best = None
        except Exception:
            dist_best = None

        results["best_match"] = {
            "id": best_row.get("id", "N/A"),
            "host_id": best_row.get("host_id", "N/A"),
            "similarity": max_sim,
            "distance_meters": dist_best,
            "description": best_row.get("description", ""),
            "neighborhood_overview": best_row.get("neighborhood_overview", ""),
        }

    # 6️⃣ 文本相似度 >= 阈值的同 host 房源
    matched_mask = all_similarities >= similarity_threshold
    matched_indices = np.where(matched_mask)[0]
    results["matched_count"] = int(len(matched_indices))

    similar_listings = []
    for idx in matched_indices:
        row = same_host_df.iloc[int(idx)]
        similar_listings.append(
            {
                "id": row.get("id", "N/A"),
                "host_id": row.get("host_id", "N/A"),
                "similarity": float(all_similarities[idx]),
                "latitude": row.get("latitude", None),
                "longitude": row.get("longitude", None),
                "description": row.get("description", ""),
                "neighborhood_overview": row.get("neighborhood_overview", ""),
            }
        )
    results["similar_listings"] = similar_listings

    # 7️⃣ 最终：同一 host 内，同时满足“近距离 + 文本相似”的，才算潜在幽灵酒店
    results["description_match"] = results["matched_count"] > 0
    results["is_potential_ghost_hotel"] = (
        results["geographic_match"] and results["description_match"]
    )

    return results

def main():
    # 先加载数据，避免在 sidebar / 主体重复加载
    listings_df = load_ghost_hotel_data()

    st.title("🏨 幽灵酒店检测平台")
    st.markdown("""
本平台基于 **Host ID 匹配**、**地理位置距离** 与 **文本相似度分析** 来识别潜在的“幽灵酒店”房源（集中经营、批量式短租单元等）。

**检测逻辑如下：**

1. **Host ID 匹配**  
   系统首先从 Airbnb 数据中筛选与输入房源具有相同 Host ID 的所有房源，仅在同一 Host 范围内进行后续检测。

2. **距离条件（默认 200 米）**  
   在同一 Host 的房源中，若存在与输入房源的测地线距离  
   **小于等于设定阈值** 的房源，则认为可能由同一房东集中经营。

3. **文本相似度条件（默认阈值 0.5）**  
   对于同一 Host 且距离在阈值内的房源，系统会计算描述文本（及可选的社区描述）的相似度。  
   若相似度 **大于等于阈值**，则认为房源内容高度相似，可能存在批量复制文本行为。

**最终判定：**  
只要满足以下三个条件之一即会被标记为潜在幽灵酒店房源：同一 Host ID + 距离在阈值范围内 + 文本相似度达到阈值
    """)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        distance_threshold = st.slider("地理距离阈值（米）", 50, 500, 200, 50)
        similarity_threshold = st.slider("文本相似度阈值", 0.0, 1.0, 0.5, 0.05)

        st.markdown("---")
        st.header("📊 数据状态")
        if listings_df is not None:
            st.success(f"✅ 已加载 {len(listings_df):,} 条 Airbnb 房源记录")
        else:
            st.warning("⚠️ 未找到 Airbnb 房源数据文件")
    
    # 使用说明（更新为“任一条件”版本）
    with st.expander("📖 使用说明"):
        st.markdown(f"""
### 幽灵酒店检测平台（Ghost Hotel Detection Platform）

本平台基于 Airbnb 房源数据，通过空间分析与文本相似度分析对潜在“幽灵酒店”（非法集中式短租）进行识别。平台支持单条检测与批量 CSV 检测，并提供地图可视化与详尽的检测结果说明。

1. 检测逻辑说明
平台基于以下三类特征进行综合判断，只要满足任意一个条件，即会被标记为潜在可疑房源。

1.1 Host ID + 空间距离匹配判断
系统首先从 Airbnb 数据集中筛选与输入房源具有相同 Host ID 的所有房源，其后所有判断均基于同一 Host 的房源列表。
然后，检查该 Host 名下的房源中是否存在与输入房源的地理测地线距离小于等于指定阈值（默认 200 米）的其他房源。
大量高密度房源（同一 Host）通常代表潜在集中经营。

1.2 文本相似度判断

对于同一 Host 且距离小于等于阈值的房源，平台会: 
1）计算输入房源描述（description）与对方房源描述的文本相似度

2）同时比较社区概述（neighborhood_overview）

3）两者取最大值作为该房源的最终相似度

若1）和2）的相似度大于等于设定阈值（默认 0.5），则说明房源内容高度相似，可能存在批量复制粘贴问题。

1.4 最终判定

如果满足以下条件: Host ID 相同+距离在阈值范围内+文本相似度不低于阈值，则该房源将被标记为潜在幽灵酒店房源。

---

### 2. 输入内容说明

平台提供两种检测方式：单条检测与批量上传检测。

2.1 单条房源检测
用户需提供以下信息：
Host ID：房东的标识符，用于筛选同房东房源
Latitude：输入房源的纬度
Longitude：输入房源的经度
Description：房源描述文本，用于文本相似度计算
Neighborhood Overview：社区概述文本（可为空）

所有文本将经过分词和 TF-IDF 分析，用于计算相似度。

2.2 批量 CSV 检测
上传的 CSV 需包含以下列：
必需列
id（房源唯一标识符）、host_id（Host ID）、latitude（纬度）、longitude（经度）、description（房源描述）
可选列：
neighborhood_overview（社区概述）

---

### 3. 检测结果说明

系统会基于用户设定的空间阈值与相似度阈值提供检测结果，包括：

3.1 总体判断

输出是否属于潜在幽灵酒店房源，包括：

是否存在同一 Host 的其他房源

是否与这些房源距离小于设定阈值

是否与这些房源存在高文本相似度

3.2 关键指标

输出如下核心指标：

距离阈值内的房源数量

文本相似度不低于设定阈值的房源数量

输入房源的最高文本相似度

输入房源与同 Host 最近房源的距离

这些指标用于衡量该房源是否具备集中管理、复制粘贴模板文本等风险特征。

3.3 详细房源列表

系统还将展示：

与输入房源距离在阈值内的 Airbnb 房源列表

文本相似度达到阈值的高相似度房源列表

文本最相似的房源及其完整描述内容

3.4 批量检测结果

批量检测将生成一个包含如下列的新 CSV 文件：
- id（房源 ID）
- latitude（输入纬度）
- longitude（输入经度）
- similarity_score（输入房源的最高文本相似度）
- geographic_match（是否存在距离内房源）
- description_match（是否存在文本相似度满足阈值的房源）
- is_potential_ghost_hotel（是否最终被标记为潜在可疑房源）
- nearby_count（距离内房源数量）
- matched_count（文本相似度达标的房源数量）
""")
    
    # 主输入区域
    st.header("📝 输入待检测房源信息")

    col_form, col_map = st.columns([1.2, 1])  # 左宽右窄一点

    with col_form:
        host_id_input = st.text_input(
        "Host ID",
        value="",
        help="请输入该房源对应的 Airbnb host_id（用于在同一房东名下做检测）",
        key="host_id_input",)

        latitude = st.number_input(
            "纬度 (Latitude)",
            min_value=-90.0,
            max_value=90.0,
            value=51.5074,
            format="%.6f",
            help="例如：51.5074（伦敦市中心）"
        )
        longitude = st.number_input(
            "经度 (Longitude)",
            min_value=-180.0,
            max_value=180.0,
            value=-0.1278,
            format="%.6f",
            help="例如：-0.1278（伦敦市中心）"
        )
        description = st.text_area(
            "房源描述 (Description)",
            height=150,
            placeholder="请输入房源的详细描述...",
            help="这是检测文本相似度的主要依据",
            key="description_input"
        )

        neighborhood_overview = st.text_area(
            "社区概述 (Neighborhood Overview)",
            height=100,
            placeholder="请输入社区或周边环境的描述...（可选）",
            help="可选字段，用于辅助检测",
            key="overview_input"
        )

    with col_map:
        st.markdown("**📍 伦敦 Airbnb 空间分布 & 当前房源位置**")
        layers = []

        # 红色：已知 Airbnb 房源
        if listings_df is not None and not listings_df.empty:
            gh_df = listings_df[['latitude', 'longitude']].dropna()
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=gh_df,
                    get_position='[longitude, latitude]',
                    get_radius=15,
                    get_fill_color=[255, 0, 0, 150],  # 红色
                    pickable=True,
                    tooltip={"text": "红点：现有 Airbnb 房源"},
                )
            )

        # 白色：当前输入的候选房源
        current_point = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        layers.append(
            pdk.Layer(
        "ScatterplotLayer",
        data=current_point,
        get_position='[lon, lat]',
        get_radius=50,
        radius_min_pixels=8, 
        get_fill_color=[255, 0, 0, 300],
        pickable=True,
        tooltip={"text": "白点：待检测房源"},
            )
        )

        view_state = pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=11,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=view_state,
                layers=layers,
                map_style=None,
            )
        )

    # 检测按钮
    if st.button("🔍 开始检测", type="primary", use_container_width=True):
        if listings_df is None or listings_df.empty:
            st.error("❌ 当前未加载 Airbnb 房源数据，无法进行检测。")
            return
        if not host_id_input.strip():
            st.error("❌ 请先输入 Host ID！")
            return

        if not description.strip():
            st.error("❌ 请至少输入房源描述信息！")
            return

        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            st.error("❌ 请输入有效的经纬度坐标！")
            return

        # 显示加载状态
        with st.spinner("正在检测中，请稍候..."):
            results = detect_ghost_hotel(
        host_id=host_id_input,
        description=description,
        neighborhood_overview=neighborhood_overview if neighborhood_overview.strip() else "",
        latitude=latitude,
        longitude=longitude,
        listings_df=listings_df,
        distance_threshold=distance_threshold,
        similarity_threshold=similarity_threshold,
            )
        
        # 显示结果
        st.header("🔍 检测结果")

        if results['is_potential_ghost_hotel']:
            triggered_conditions = []
            if results['geographic_match']:
                triggered_conditions.append(
                    f"🗺️ 与至少一条 Airbnb 房源的距离 ≤ {distance_threshold} 米"
                    f"且同一 Host ID：{host_id_input}"
                )
            if results['description_match']:
                triggered_conditions.append(
                    f"📝 与至少一条 Airbnb 房源的文本相似度 ≥ {similarity_threshold:.0%}"
                )
            
            st.error("🚨 检测结果：**存在潜在有问题房源特征**")
            st.markdown(
                "<div style='background-color:#ffebee;padding:16px;border-radius:10px;border-left:5px solid #f44336;'>"
                "<h4>触发条件如下：</h4>"
                "<ul>" +
                "".join([f"<li>{c}</li>" for c in triggered_conditions]) +
                "</ul>"
                f"<p>本次检测的最高文本相似度为：<b>{results['similarity_score']:.2%}</b></p>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.success("✅ 检测结果：在当前阈值下未发现明显的可疑特征")
            st.info(
                f"该房源在 {distance_threshold:.0f} 米范围内没有发现现有 Airbnb 房源，"
                f"且同一 Host ID：{host_id_input}，"
                f"且与现有房源的最高文本相似度为 {results['similarity_score']:.2%}（低于设定阈值 {similarity_threshold:.0%}）。"
            )

        # 详细信息
        with st.expander("📋 查看详细信息", expanded=results['is_potential_ghost_hotel']):
        # 先算一下最近距离（如果有附近房源的话）
            if results['nearby_listings']:
                nearest_distance = min(h.get("distance_meters", float("inf")) for h in results['nearby_listings'])
                # 防御：如果都是 inf 或缺失
                if nearest_distance == float("inf"):
                    nearest_distance = None
            else:
                nearest_distance = None

            # 数量信息
            col1, col2 = st.columns(2)
            with col1:
                st.metric("距离阈值内的房源数量", len(results['nearby_listings']))
            with col2:
                st.metric(f"文本相似度 ≥ {similarity_threshold:.0%}的房源数量", results['matched_count'])

            # 相似度 & 最近距离
            col1, col2 = st.columns(2)
            with col1:
                st.metric("最高文本相似度", f"{results['similarity_score']:.2%}")
            with col2:
                if nearest_distance is not None:
                    st.metric("与输入房源最近的房源之间的距离", f"{nearest_distance:.2f} m")
                else:
                    st.metric("与输入房源最近的房源之间的距离", "N/A")
            
            # 展示“距离阈值内”的房源
            if results['nearby_listings']:
                st.subheader(f"📍 距离{distance_threshold:.0f} 米内的 Airbnb 房源（同一 Host ID：{host_id_input}）")
                for i, hotel in enumerate(results['nearby_listings'][:10], 1):
                    with st.container():
                        st.markdown(f"**#{i} 距离：{hotel['distance_meters']} m**")
                        st.markdown(f"**ID：`{hotel.get('id', 'N/A')}`（同一 Host ID：{host_id_input}）**")
                        st.markdown(f"**Host ID：`{hotel.get('host_id', 'N/A')}`（同一 Host ID：{host_id_input}）**")
                        if hotel.get('description') and str(hotel['description']) != 'N/A':
                            st.text(f"Description：{str(hotel['description'])[:200]}...")
                            st.text(f"Neighborhood Overview：{str(hotel['neighborhood_overview'])[:200]}...")
                        st.markdown("---")
            
            # 展示“文本相似度 ≥ 阈值”的房源
            if results['similar_listings']:
                st.subheader(f"📝 文本相似度 ≥ {similarity_threshold:.0%}的 Airbnb 房源（同一 Host ID：{host_id_input}）")
                for i, hotel in enumerate(results['similar_listings'][:10], 1):
                    with st.container():
                        st.markdown(
                            f"**#{i} 相似度：{hotel['similarity']:.2%}** | ID：{hotel.get('id', 'N/A')} | Host ID：{hotel.get('host_id', 'N/A')}")
                        if hotel.get('description'):
                            st.text(f"Description：{str(hotel['description'])[:200]}...")
                            st.text(f"Neighborhood Overview：{str(hotel['neighborhood_overview'])[:200]}...")
                        st.markdown("---")

            # 显示“最相似”的那一条
            if results.get('best_match'):
                bm = results['best_match']
                st.subheader("⭐ 文本上最相似的 Airbnb 房源")
                st.markdown(
                    f"- ID：`{bm.get('id', 'N/A')}`\n"
                    f"- Host ID：`{bm.get('host_id', 'N/A')}`\n"
                    f"- 文本相似度：**{bm.get('similarity', 0.0):.2%}**"
                )
                if bm.get('description'):
                    st.text(f"描述：{str(bm['description'])[:300]}...")
                    st.text(f"Neighborhood Overview：{str(bm['neighborhood_overview'])[:300]}...")
    
    # ===================== 批量 CSV 检测 =====================
    st.header("📂 批量 CSV 房源检测")

    st.markdown(
        "你可以上传一个包含 **房源ID、Host ID、经纬度、房源描述、社区概述** 的 CSV 文件，"
        "系统会对每一条记录执行与上面相同的检测逻辑（同一 Host + 距离 + 文本相似度），"
        "并标注是否为潜在有问题房源。"
    )

    uploaded_file = st.file_uploader("上传待检测房源 CSV 文件", type=["csv"])

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ 无法读取该 CSV 文件：{e}")
            user_df = None

        if user_df is not None:
            st.write("✅ 已成功读取上传文件")

            # --- 根据列名智能猜测 ---
            def guess_col(candidates):
                candidates = {c.lower() for c in candidates}
                for col in user_df.columns:
                    if col.lower() in candidates:
                        return col
                return None

            id_guess = guess_col({"id", "listing_id", "airbnb_id"})
            host_guess = guess_col({"host_id", "host", "owner_id"})
            lat_guess = guess_col({"latitude", "lat", "y"})
            lon_guess = guess_col({"longitude", "lon", "lng", "x"})
            desc_guess = guess_col({"description", "desc", "listing_description"})
            ov_guess = guess_col({
                "neighborhood_overview", "neighbourhood_overview",
                "neighborhood", "neighbourhood", "area_description"
            })

            cols = list(user_df.columns)

            def _default_index(col_name):
                if col_name in cols:
                    return cols.index(col_name)
                return 0

            st.subheader("⚙️ 输入房源信息设置")
            col_a, col_b = st.columns(2)
            with col_a:
                id_col = st.selectbox("房源 ID 列", options=cols, index=_default_index(id_guess))
                host_col = st.selectbox("Host ID 列", options=cols, index=_default_index(host_guess))
                lat_col = st.selectbox("纬度列 (Latitude)", options=cols, index=_default_index(lat_guess))
            with col_b:
                lon_col = st.selectbox("经度列 (Longitude)", options=cols, index=_default_index(lon_guess))
                desc_col = st.selectbox("房源描述列 (Description)", options=cols, index=_default_index(desc_guess))
                ov_col = st.selectbox(
                    "社区概述列 (Neighborhood Overview，可选)",
                    options=["<无此列>"] + cols,
                    index=(0 if ov_guess is None else _default_index(ov_guess) + 1)
                )

            # 开始批量检测按钮
            if st.button("🚀 对上传 CSV 执行批量检测", type="primary", use_container_width=True):
                if listings_df is None or listings_df.empty:
                    st.error("❌ 当前基准 Airbnb 数据为空，无法进行检测。")
                else:
                    result_rows = []
                    invalid_rows = 0
                    with st.spinner("正在对上传文件中的房源逐条检测，请稍候..."):

                        for _, row in user_df.iterrows():
                            # 1) Host ID
                            host_val = row.get(host_col, None)
                            if pd.isna(host_val) or str(host_val).strip() == "":
                                invalid_rows += 1
                                continue
                            host_val = str(host_val).strip()

                            # 2) 经纬度
                            try:
                                lat_val = float(row[lat_col])
                                lon_val = float(row[lon_col])
                            except Exception:
                                invalid_rows += 1
                                continue

                            # 3) 文本字段
                            desc_raw = row.get(desc_col, "")
                            desc_val = str(desc_raw) if pd.notna(desc_raw) else ""
                            if ov_col == "<无此列>":
                                ov_val = ""
                            else:
                                ov_raw = row.get(ov_col, "")
                                ov_val = str(ov_raw) if pd.notna(ov_raw) else ""

                            # 4) 调用新的按 host_id 的检测逻辑
                            det = detect_ghost_hotel(
                                description=desc_val,
                                neighborhood_overview=ov_val,
                                latitude=lat_val,
                                longitude=lon_val,
                                host_id=host_val, 
                                listings_df=listings_df,
                                distance_threshold=distance_threshold,
                                similarity_threshold=similarity_threshold,
                            )

                            result_rows.append({
                                "id": row[id_col],
                                "host_id": host_val,
                                "latitude": lat_val,
                                "longitude": lon_val,
                                "similarity_score": det["similarity_score"],
                                "geographic_match": det["geographic_match"],
                                "description_match": det["description_match"],
                                "is_potential_ghost_hotel": det["is_potential_ghost_hotel"],
                                "nearby_count": len(det["nearby_listings"]),
                                "matched_count": det["matched_count"],
                            })

                    if result_rows:
                        batch_result_df = pd.DataFrame(result_rows)
                        st.subheader("📊 批量检测结果预览")
                        st.dataframe(batch_result_df)

                        # 总结信息
                        total = len(batch_result_df)
                        flagged = int(batch_result_df["is_potential_ghost_hotel"].sum())
                        st.markdown(
                            f"- 总检测房源数：**{total}**\n"
                            f"- 被标记为潜在有问题房源的数量：**{flagged}**"
                        )
                        if invalid_rows > 0:
                            st.warning(f"有 {invalid_rows} 行由于 Host ID / 经纬度缺失或格式问题被跳过。")

                        # 提供下载按钮
                        csv_bytes = batch_result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ 下载批量检测结果 CSV",
                            data=csv_bytes,
                            file_name="ghost_hotel_batch_detection_results.csv",
                            mime="text/csv",
                        )
                    else:
                        st.info("未生成任何检测结果，可能是上传文件中没有有效的 Host ID 或经纬度。")

    # Airbnb 名单表格
    if listings_df is not None and not listings_df.empty:
        with st.expander("📋 伦敦 Airbnb 房源样本名单", expanded=False):
            st.markdown("以下为样本名单（最多显示前 200 条），包含位置与部分文本信息，方便快速浏览与校验。")
            st.dataframe(
                listings_df[['id','host_id','latitude', 'longitude', 'room_type','description', 'neighborhood_overview','number_of_reviews']].head(200)
            )

if __name__ == "__main__":
    main()
