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

# ===================== 文本相似度相关函数 =====================
import re

def simple_tokenize(text):
    """
    一个不依赖 NLTK 的安全分词器：
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
                    'distance_meters': round(distance, 2),
                    'description': row.get('description', 'N/A'),
                    'neighborhood_overview': row.get('neighborhood_overview', 'N/A')
                })
        except (ValueError, KeyError):
            continue

    return nearby_listings


# ===================== 核心检测逻辑 =====================

def detect_ghost_hotel(
    description: str,
    neighborhood_overview: str,
    latitude: float,
    longitude: float,
    listings_df: pd.DataFrame,
    distance_threshold: float = 200,
    similarity_threshold: float = 0.5
):
    """
    检测房源是否为潜在的有问题房源：
    ✅ 条件一：与任意已有 Airbnb listing 的直线距离 <= distance_threshold（默认 200m）
    ✅ 条件二：与任意已有 Airbnb listing 的文本相似度 >= similarity_threshold（默认 0.5）

    只要满足「任意一条」，就视为潜在有问题房源。
    返回：
    - is_potential_ghost_hotel: 是否存在空间或文本上的可疑匹配
    - geographic_match: 是否满足空间条件（存在 200m 内 listing）
    - description_match: 是否满足文本相似度条件（存在相似度 >= 阈值的 listing）
    - similarity_score: 所有 listing 中的最高文本相似度（scalar）
    - nearby_listings: 所有 200m 内的 listing 详情
    - similar_listings: 所有文本相似度 >= 阈值的 listing 详情
    - best_match: 文本相似度最高的那一条 listing（包含 id、相似度等）
    """

    results = {
        'is_potential_ghost_hotel': False,
        'geographic_match': False,
        'description_match': False,
        'similarity_score': 0.0,
        'nearby_listings': [],
        'similar_listings': [],
        'best_match': None,
        'details': {}
    }

    if listings_df is None or listings_df.empty:
        results['details'] = {
            'warning': '无法加载 Airbnb 基础数据，仅进行文本分析不可行',
            'message': '请确保数据文件存在或网络连接正常'
        }
        return results

    # ---------- 1. 空间条件：200m 内 listing ----------
    nearby = check_geographic_proximity(
        latitude,
        longitude,
        listings_df,
        threshold_meters=distance_threshold
    )
    results['nearby_listings'] = nearby
    if nearby:
        results['geographic_match'] = True

    # ---------- 2. 文本条件：相似度 >= similarity_threshold ----------
    cand_desc = listings_df['description'].fillna("").astype(str).tolist() \
        if 'description' in listings_df.columns else []
    cand_ov = listings_df['neighborhood_overview'].fillna("").astype(str).tolist() \
        if 'neighborhood_overview' in listings_df.columns else []

    best_sim, best_idx, sims_desc = best_similarity_with_candidates(
        description,
        neighborhood_overview,
        cand_desc,
        cand_ov
    )
    results['similarity_score'] = best_sim

    # 找出「文本相似度 >= 阈值」的所有 listing
    similar_listings = []
    if sims_desc is not None and len(sims_desc) == len(listings_df):
        for i, sim_val in enumerate(sims_desc):
            if float(sim_val) >= similarity_threshold:
                row = listings_df.iloc[i]
                similar_listings.append({
                    'id': row.get('id', 'N/A'),
                    'similarity': float(sim_val),
                    'description': row.get('description', ''),
                    'neighborhood_overview': row.get('neighborhood_overview', '')
                })

    if similar_listings:
        results['description_match'] = True
        # 可以按相似度排序，方便前端展示
        similar_listings = sorted(similar_listings, key=lambda x: x['similarity'], reverse=True)

    results['similar_listings'] = similar_listings

    # 记录文本上“最像”的那一条
    if best_idx is not None and 0 <= best_idx < len(listings_df):
        row = listings_df.iloc[best_idx]
        results['best_match'] = {
            'id': row.get('id', 'N/A'),
            'similarity': best_sim,
            'description': row.get('description', ''),
            'neighborhood_overview': row.get('neighborhood_overview', '')
        }

    # ---------- 3. 总体判定：空间 OR 文本 任一满足即可 ----------
    if results['geographic_match'] or results['description_match']:
        results['is_potential_ghost_hotel'] = True

    return results




def main():
    # 先加载数据，避免在 sidebar / 主体重复加载
    listings_df = load_ghost_hotel_data()

    st.title("🏨 幽灵酒店检测平台")
    st.markdown("""
    本平台基于 **文本相似度** 和 **地理距离** 检测潜在的有问题房源（幽灵酒店 / 非法短租集群）。

    **当前判定规则：只要满足以下任一条件，即视为“潜在有问题房源”**：

    1. 🗺️ 地理条件：与任意已有 Airbnb 房源的直线距离 **小于等于设定阈值**（默认 200 米）
    2. 📝 文本条件：与任意已有 Airbnb 房源的 **描述文本相似度** 高于设定阈值（默认 0.5）

    只要满足其中一条，系统都会将该房源标记为“潜在有问题房源”，并列出对应的匹配结果。
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

    # 主输入区域
    st.header("📝 输入待检测房源信息")

    col_form, col_map = st.columns([1.2, 1])  # 左宽右窄一点

    with col_form:
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
                    get_radius=30,
                    get_fill_color=[255, 0, 0, 150],  # 红色
                    pickable=True,
                )
            )

        # 白色：当前输入的候选房源
        current_point = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=current_point,
                get_position='[lon, lat]',
                get_radius=80,
                get_fill_color=[255, 255, 255, 255],  # 白色
                get_line_color=[0, 0, 0],
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
                tooltip={"text": "红点：现有 Airbnb 房源\n白点：当前候选房源"},
            )
        )

    # 检测按钮
    if st.button("🔍 开始检测", type="primary", use_container_width=True):
        if listings_df is None or listings_df.empty:
            st.error("❌ 当前未加载 Airbnb 房源数据，无法进行检测。")
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
                description,
                neighborhood_overview if neighborhood_overview.strip() else "",
                latitude,
                longitude,
                listings_df,
                distance_threshold=distance_threshold,
                similarity_threshold=similarity_threshold
            )

        # 显示结果
        st.header("📊 检测结果")

        if results['is_potential_ghost_hotel']:
            triggered_conditions = []
            if results['geographic_match']:
                triggered_conditions.append(f"🗺️ 与至少一条 Airbnb 房源的距离 ≤ {distance_threshold} 米")
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
                f"该房源在 {distance_threshold} 米范围内没有发现现有 Airbnb 房源，"
                f"且与现有房源的最高文本相似度为 {results['similarity_score']:.2%}（低于设定阈值 {similarity_threshold:.0%}）。"
            )

        # 详细信息
        with st.expander("📋 查看详细信息", expanded=results['is_potential_ghost_hotel']):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("最高文本相似度", f"{results['similarity_score']:.2%}")
                st.metric("距离阈值内的房源数量", len(results['nearby_listings']))
            with col2:
                st.metric("文本相似度 ≥ 阈值的房源数量", len(results['similar_listings']))
                st.metric("地理位置匹配", "是" if results['geographic_match'] else "否")

            # 展示“距离阈值内”的房源
            if results['nearby_listings']:
                st.subheader("📍 距离在阈值内的 Airbnb 房源")
                for i, hotel in enumerate(results['nearby_listings'][:10], 1):  # 最多显示 10 条
                    with st.container():
                        st.markdown(f"**#{i} 距离：{hotel['distance_meters']} 米**")
                        if hotel.get('description') and str(hotel['description']) != 'N/A':
                            st.text(f"描述：{str(hotel['description'])[:200]}...")
                        st.markdown("---")

            # 展示“文本相似度 ≥ 阈值”的房源
            if results['similar_listings']:
                st.subheader("📝 文本相似度 ≥ 阈值的 Airbnb 房源")
                for i, hotel in enumerate(results['similar_listings'][:10], 1):
                    with st.container():
                        st.markdown(
                            f"**#{i} 相似度：{hotel['similarity']:.2%}**  | ID：{hotel.get('id', 'N/A')}"
                        )
                        if hotel.get('description'):
                            st.text(f"描述：{str(hotel['description'])[:200]}...")
                        st.markdown("---")

            # 显示“最相似”的那一条
            if results.get('best_match'):
                bm = results['best_match']
                st.subheader("⭐ 文本上最相似的 Airbnb 房源")
                st.markdown(
                    f"- ID：`{bm.get('id', 'N/A')}`\n"
                    f"- 文本相似度：**{bm.get('similarity', 0.0):.2%}**"
                )
                if bm.get('description'):
                    st.text(f"描述：{str(bm['description'])[:300]}...")

    # Airbnb 名单表格
    if listings_df is not None and not listings_df.empty:
        with st.expander("📋 伦敦 Airbnb 房源样本名单", expanded=False):
            st.markdown("以下为样本名单（最多显示前 200 条），包含位置与部分文本信息，方便快速浏览与校验。")
            st.dataframe(
                listings_df[['id', 'latitude', 'longitude', 'description', 'neighborhood_overview']].head(200)
            )

    # 使用说明（更新为“任一条件”版本）
    with st.expander("📖 使用说明"):
        st.markdown(f"""
1. **输入房源信息**：
   - 在左侧输入房源的经纬度坐标（Latitude / Longitude）
   - 在主区域中输入房源的详细描述 **(Description，必填)**  
   - 如有需要，可补充社区概述 **(Neighborhood Overview，可选)**，有助于提高文本相似度检测的准确性

2. **点击检测**：
   - 系统会执行两类分析：
     1. **地理距离检测**：计算该房源与所有现有 Airbnb 房源之间的直线距离，找出距离小于等于你在侧边栏设置的阈值（当前：**{distance_threshold} 米**）的房源
     2. **文本相似度检测**：基于 TF-IDF 与相似度计算，分析当前房源描述与所有 Airbnb 房源描述/社区概述之间的文本相似度

3. **判定逻辑**（当前版本）：
   - 若存在任意一条 Airbnb 房源满足：
     - 🗺️ 与该房源的直线距离 ≤ **{distance_threshold} 米**，**或**
     - 📝 与该房源的文本相似度 ≥ **{similarity_threshold:.0%}**
     
     则该房源会被标记为 **「潜在有问题房源」**。

4. **结果查看**：
   - 在「📋 查看详细信息」中，你可以看到：
     - 最高文本相似度
     - 距离阈值内的房源数量
     - 文本相似度 ≥ 阈值的房源数量
     - 距离在阈值内的具体房源列表
     - 文本上最相似的 Airbnb 房源及其描述片段

5. **阈值调整建议**：
   - 若希望 **更敏感（宁可多报）**：
     - 可以适当 **增大** 地理距离阈值（例如 300–400 米）
     - 或 **降低** 文本相似度阈值（例如 0.4）
   - 若希望 **更保守（宁可漏报）**：
     - 可以适当 **减小** 地理距离阈值
     - 或 **提高** 文本相似度阈值（例如 0.6）

6. **局限性说明**：
   - 本平台基于已有的 Airbnb 数据，在数据覆盖不全或房源信息不完整时，可能产生漏检或误判；
   - 文本相似度依赖于房东的描述风格，模板化描述可能导致相似度偏高；
   - 地理距离为平面近似，并不能区分同一栋楼内不同法律属性的单位。
        """)

if __name__ == "__main__":
    main()
