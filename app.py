import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STORAGE_FILE = "movie_storage.json"

def load_local_data():
    """로컬 파일에서 영화 보관함 데이터 불러오기 (구버전 호환)"""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                df = pd.DataFrame(data.get('movies', []))
                
                for col in ["id", "title", "genre", "overview", "features", "poster"]:
                    if col not in df.columns:
                        df[col] = "" if col != "id" else 0
                        
                st.session_state['custom_movies'] = df
                return
        except Exception:
            pass
            
    st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "genre", "overview", "features", "poster"])

def save_local_data():
    """영화 보관함 데이터 저장"""
    data = {
        'movies': st.session_state['custom_movies'].to_dict(orient='records')
    }
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 1. 페이지 설정 및 세션 초기화
st.set_page_config(page_title="3가지 알고리즘 영화 추천 시스템", layout="wide")

TMDB_API_KEY = "71f533a402be87b54aea626f2b1ef567" 

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'
if 'search_result' not in st.session_state:
    st.session_state['search_result'] = None

if 'custom_movies' not in st.session_state:
    load_local_data()

if 'custom_ratings' not in st.session_state:
    st.session_state['custom_ratings'] = {'다른 관객1': {}, '다른 관객2': {}, '나': {}}

for title in st.session_state['custom_movies']['title'].tolist():
    for user in st.session_state['custom_ratings']:
        if title not in st.session_state['custom_ratings'][user]:
            st.session_state['custom_ratings'][user][title] = np.nan

# TMDB API 검색 함수 (장르/줄거리 분리 수집)
def search_movie_tmdb(query):
    if not TMDB_API_KEY or TMDB_API_KEY == "여기에_발급받은_API키를_넣으세요":
        st.error("API 키를 등록해주세요.")
        return None
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=ko-KR"
    try:
        response = requests.get(url).json()
        if response.get('results'):
            movie_data = response['results'][0]
            movie_id = movie_data['id']
            title = movie_data['title']
            plot = movie_data['overview'] if movie_data['overview'] else "줄거리 정보 없음"
            poster_path = movie_data['poster_path']
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else ""
            
            genre_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=ko-KR"
            genre_res = requests.get(genre_url).json()
            genres = " ".join([g['name'] for g in genre_res.get('genres', [])])
            if not genres:
                genres = "장르 정보 없음"
            
            return {
                "title": title, 
                "genre": genres, 
                "overview": plot, 
                "features": f"{genres} {plot}", 
                "poster": poster_url
            }
    except Exception as e:
        st.error(f"API 호출 오류: {e}")
    return None

# ==========================================
# 📄 화면 0: 메인 페이지 (완전 랜덤 5종 생성)
# ==========================================
if st.session_state['current_page'] == 'main':
    st.title("🎬 3가지 알고리즘으로 만든 영화 추천 프로그램")
    st.subheader("콘텐츠 기반 · 협업 필터링 · 하이브리드 모델을 활용한 영화 플랫폼")
    st.write("실제 TMDB 영화 API 데이터와 사용자의 동적 입력 데이터를 기반으로 3가지 알고리즘을 실시간 작동 및 비교 검증합니다.")
    st.markdown("---")
    
    st.subheader("📥 API 연동 실시간 영화 등록")
    
    # 🎯 극적 무작위 추출 버튼 (누를 때마다 달라짐)
    if st.button("🎲 발표용 극적 데이터셋 랜덤 생성 (극단적 장르 5종 무작위 추출)", type="primary", use_container_width=True):
        # 5개 극단적 장르 그룹
        group1 = ["토이 스토리", "주토피아", "인사이드 아웃", "슈렉"]
        group2 = ["인터스텔라", "인셉션", "마션", "테넷"]
        group3 = ["곤지암", "겟 아웃", "컨저링", "파묘"]
        group4 = ["라라랜드", "비포 선라이즈", "어바웃 타임", "뷰티 인사이드"]
        group5 = ["어벤져스", "다크 나이트", "범죄도시", "매트릭스"]
        
        # 각 그룹별로 1개씩 무작위 선정
        selected_queries = [
            random.choice(group1),
            random.choice(group2),
            random.choice(group3),
            random.choice(group4),
            random.choice(group5)
        ]
        
        new_movies = []
        with st.spinner(f"랜덤 추출된 영화 5종 ({', '.join(selected_queries)}) 데이터를 TMDB API에서 검색 중..."):
            for q in selected_queries:
                m_info = search_movie_tmdb(q)
                if m_info:
                    new_movies.append(m_info)
        
        if new_movies:
            df_new = pd.DataFrame(new_movies)
            df_new['id'] = range(1, len(df_new) + 1)
            st.session_state['custom_movies'] = df_new
            
            t = [m['title'] for m in new_movies]
            # 추출된 5개 영화에 대한 극단적인 유저 평점 자동 구성
            st.session_state['custom_ratings'] = {
                '다른 관객1': {t[0]: 5.0, t[1]: 1.0, t[2]: 1.0, t[3]: 5.0, t[4]: 4.0},
                '다른 관객2': {t[0]: 1.0, t[1]: 5.0, t[2]: 5.0, t[3]: 2.0, t[4]: 5.0},
                '나': {t[0]: 5.0, t[1]: np.nan, t[2]: np.nan, t[3]: 4.0, t[4]: np.nan}
            }
            save_local_data()
            st.success(f"🎉 극적 랜덤 5종 ({', '.join(t)}) 영화 보관함 등록 완료!")
            st.rerun()

    st.write("")
    
    col_in1, col_in2 = st.columns([1, 2])
    
    with col_in1:
        search_query = st.text_input("영화 제목 개별 검색 (한국어/영어):", placeholder="예: 토이 스토리 또는 주토피아")
        
        if st.button("🔍 영화 검색하기", use_container_width=True):
            if search_query:
                with st.spinner("TMDB API에서 영화 데이터를 찾는 중..."):
                    result = search_movie_tmdb(search_query)
                    if result:
                        st.session_state['search_result'] = result
                    else:
                        st.error("영화 데이터를 찾지 못했습니다. 제목을 정확히 입력해 주세요.")
                        st.session_state['search_result'] = None
            else:
                st.error("검색할 영화 제목을 입력하세요.")
        
        if st.session_state['search_result']:
            res = st.session_state['search_result']
            st.markdown("---")
            st.write("### 🔍 검색된 영화 확인")
            
            preview_col1, preview_col2 = st.columns([1, 2])
            with preview_col1:
                if res['poster']:
                    st.image(res['poster'], width=120)
            with preview_col2:
                st.write(f"#### **{res['title']}**")
                st.caption(f"**데이터 내용:** {res['features'][:120]}...")
            
            if st.button(f"➕ [{res['title']}]을 영화 보관함에 등록하기", type="primary", use_container_width=True):
                if res['title'] not in st.session_state['custom_movies']['title'].tolist():
                    new_id = len(st.session_state['custom_movies']) + 1
                    new_row = pd.DataFrame([{
                        "id": new_id, 
                        "title": res['title'], 
                        "genre": res['genre'],
                        "overview": res['overview'],
                        "features": res['features'], 
                        "poster": res['poster']
                    }])
                    st.session_state['custom_movies'] = pd.concat([st.session_state['custom_movies'], new_row], ignore_index=True)
                    
                    for user in st.session_state['custom_ratings']:
                        st.session_state['custom_ratings'][user][res['title']] = np.nan
                    
                    save_local_data()
                    st.success(f"🎉 [{res['title']}] 영화 보관함 등록 성공! (창을 닫아도 보존됩니다)")
                    st.session_state['search_result'] = None
                    st.rerun()
                else:
                    st.warning("⚠️ 이미 영화 보관함에 등록된 영화입니다.")
                
    with col_in2:
        st.write("**현재 API로 구축된 영화 보관함:**")
        if len(st.session_state['custom_movies']) == 0:
            st.info("아직 등록된 영화가 없습니다.")
        else:
            movies_list = st.session_state['custom_movies']
            cols = st.columns(min(len(movies_list), 4))
            
            for idx, row in movies_list.iterrows():
                with cols[idx % 4]:
                    if row['poster']:
                        st.image(row['poster'], width=100)
                    st.caption(f"**{row['title']}**")
                    
                    if st.button("❌ 삭제", key=f"del_{row['title']}_{idx}", use_container_width=True, type="secondary"):
                        st.session_state['custom_movies'] = st.session_state['custom_movies'][st.session_state['custom_movies']['title'] != row['title']].reset_index(drop=True)
                        
                        for user in st.session_state['custom_ratings']:
                            if row['title'] in st.session_state['custom_ratings'][user]:
                                del st.session_state['custom_ratings'][user][row['title']]
                        
                        save_local_data()
                        st.rerun()
            
            st.write("")
            if st.button("🗑️ 영화 보관함 데이터 전체 초기화", use_container_width=True):
                st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "genre", "overview", "features", "poster"])
                st.session_state['custom_ratings'] = {'다른 관객1': {}, '다른 관객2': {}, '나': {}}
                st.session_state['search_result'] = None
                
                if os.path.exists(STORAGE_FILE):
                    os.remove(STORAGE_FILE)
                st.rerun()

    st.markdown("---")
    st.subheader("🚀 2단계: 알고리즘 프로그램 실행")
    
    disabled_btn = len(st.session_state['custom_movies']) < 2
    if disabled_btn:
        st.warning("⚠️ 영화가 최소 2개 이상 등록되어야 알고리즘 페이지가 활성화됩니다.")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 1️⃣ 콘텐츠 기반 필터링")
        st.write("API가 가져온 실제 영화 줄거리 및 장르 데이터를 분석해 추천합니다.")
        if st.button("콘텐츠 기반 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_content'
            st.rerun()
    with col2:
        st.success("### 2️⃣ 협업 필터링")
        st.write("등록한 실제 영화들에 평점을 부여해 가상의 사용자 평점 데이터를 구성합니다.")
        if st.button("협업 필터링 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_collaborative'
            st.rerun()
    with col3:
        st.warning("### 3️⃣ 하이브리드 시스템")
        st.write("줄거리/장르 점수와 사용자 평점 점수를 가중 결합하여 하이브리드 추천을 진행합니다.")
        if st.button("하이브리드 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_hybrid'
            st.rerun()


# ==========================================
# 📄 화면 1: 콘텐츠 기반 필터링 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_content':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("1️⃣ 콘텐츠 기반 필터링")
    movies_db = st.session_state['custom_movies']
    
    user_history = st.multiselect("과거에 극장에서 재밌게 본 영화를 선택하세요:", movies_db['title'].tolist())
    
    if user_history:
        try:
            tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
            tfidf_matrix = tfidf.fit_transform(movies_db['features'].fillna(''))
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            user_movie_indices = movies_db[movies_db['title'].isin(user_history)].index
            sim_scores = cosine_sim[user_movie_indices].mean(axis=0)
            
            recommend_df = movies_db.copy()
            recommend_df['similarity'] = sim_scores
            recommend_df = recommend_df[~recommend_df['title'].isin(user_history)]
            final_recommend = recommend_df.sort_values(by='similarity', ascending=False)
            
            st.subheader("🎯 줄거리 및 장르 데이터 분석을 통한 추천 결과")
            if final_recommend.empty:
                st.info("모든 영화를 선택하셨습니다! 추천할 다른 영화가 없습니다.")
            else:
                for _, row in final_recommend.iterrows():
                    c1, c2 = st.columns([1, 5])
                    with c1:
                        if row['poster']: st.image(row['poster'], width=120)
                    with c2:
                        st.write(f"### **{row['title']}** (유사도 매칭 점수: `{row['similarity']:.2f}`)")
                        st.write(f"💬 **[추천 이유 요약]:** 이 영화의 API 데이터베이스 요약본(`{row['features'][:70]}...`)이 과거 선호 장르/소재 패턴과 일치하여 추천되었습니다.")
                    st.markdown("---")
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")


# ==========================================
# 📄 화면 2: 협업 필터링 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_collaborative':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("2️⃣ 협업 필터링")
    movies_db = st.session_state['custom_movies']
    
    st.write("### 🎲 가상 유저들의 평점 기록")
    if st.button("다른 유저 평점 무작위 생성"):
        for user in ['다른 관객1', '다른 관객2']:
            st.session_state['custom_ratings'][user] = {title: np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]) for title in movies_db['title'].tolist()}
        st.rerun()
            
    st.write("### 👤 내 실제 영화 관람 평점 입력")
    my_ratings = st.session_state['custom_ratings'].get('나', {})
    
    changed = False
    for idx, row in movies_db.iterrows():
        current_val = my_ratings.get(row['title'], np.nan)
        default_idx = 0 if pd.isna(current_val) else int(current_val)
        
        score = st.selectbox(f"[{row['title']}] 영화에 내 평점은?", ["안봄(NaN)", "1", "2", "3", "4", "5"], index=default_idx, key=f"collab_rat_{row['title']}")
        new_val = np.nan if score == "안봄(NaN)" else float(score)
        
        if str(current_val) != str(new_val):
            my_ratings[row['title']] = new_val
            changed = True
        
    if changed:
        st.session_state['custom_ratings']['나'] = my_ratings
    
    ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
    st.write("#### 📊 실시간 영화 평점 현황판")
    st.dataframe(ratings_df)
    
    interaction_matrix = ratings_df.fillna(0)
    item_similarity = cosine_similarity(interaction_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    
    watched_movies = [m for m, r in my_ratings.items() if not pd.isna(r)]
    unwatched_movies = [m for m, r in my_ratings.items() if pd.isna(r)]
    
    if watched_movies and unwatched_movies:
        predictions = {}
        for movie in unwatched_movies:
            sim_sum = 0
            weighted_rating_sum = 0
            for watched_movie in watched_movies:
                if movie in item_sim_df.index and watched_movie in item_sim_df.columns:
                    sim = item_sim_df.loc[movie, watched_movie]
                    rating = my_ratings[watched_movie]
                    sim_sum += sim
                    weighted_rating_sum += (sim * rating)
            predictions[movie] = weighted_rating_sum / sim_sum if sim_sum > 0 else 0
            
        st.subheader("🎯 집단 행동 패턴 분석 기반 예상 별점 결과")
        for movie, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            target_rows = movies_db[movies_db['title'] == movie]
            if target_rows.empty: continue
            target_row = target_rows.iloc[0]
            
            c1, c2 = st.columns([1, 5])
            with c1:
                if target_row['poster']: st.image(target_row['poster'], width=100)
            with c2:
                st.write(f"### **{movie}** (예측 평점 점수: `{score:.2f}` 점)")
                if score == 0:
                    st.error("⚠️ 데이터 희소성으로 이 영화를 평가한 다른 유저 세트가 없어 연산이 제한됩니다 (콜드 스타트).")
            st.markdown("---")


# ==========================================
# 📄 화면 3: 하이브리드 추천 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_hybrid':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("3️⃣ 하이브리드 추천 시스템")
    movies_db = st.session_state['custom_movies']
    
    st.subheader("🎛️ 추천 비율 균형 조절")
    w_content = st.slider("줄거리/장르 가중치 설정 (%)", 0, 100, 50)
    w_collab = 100 - w_content
    
    target_movie = st.selectbox("추천 알고리즘의 기준점이 될 메인 영화 선택:", movies_db['title'].tolist())
    
    try:
        def safe_tfidf_sim(series_data):
            cleaned_data = series_data.fillna('').astype(str).str.strip()
            if cleaned_data.str.cat().strip() == "":
                return np.zeros((len(series_data), len(series_data)))
            try:
                tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
                tfidf_matrix = tfidf.fit_transform(cleaned_data)
                return cosine_similarity(tfidf_matrix, tfidf_matrix)
            except ValueError:
                return np.zeros((len(series_data), len(series_data)))

        cosine_sim = safe_tfidf_sim(movies_db['features'])
        
        genre_data = movies_db['genre'] if 'genre' in movies_db.columns else pd.Series(['']*len(movies_db))
        genre_sim = safe_tfidf_sim(genre_data)
        
        overview_data = movies_db['overview'] if 'overview' in movies_db.columns else pd.Series(['']*len(movies_db))
        plot_sim = safe_tfidf_sim(overview_data)
        
        movie_idx = movies_db[movies_db['title'] == target_movie].index[0]
        
        content_scores = pd.Series(cosine_sim[movie_idx], index=movies_db['title'].tolist())
        genre_scores = pd.Series(genre_sim[movie_idx], index=movies_db['title'].tolist())
        plot_scores = pd.Series(plot_sim[movie_idx], index=movies_db['title'].tolist())
        
        ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
        interaction_matrix = ratings_df.fillna(0)
        
        if target_movie in interaction_matrix.index and interaction_matrix.sum().sum() > 0:
            item_similarity = cosine_similarity(interaction_matrix)
            item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
            collab_scores = item_sim_df[target_movie]
        else:
            collab_scores = pd.Series(0.0, index=movies_db['title'].tolist())
            
        hybrid_df = pd.DataFrame({
            'content': content_scores,
            'genre': genre_scores,
            'plot': plot_scores,
            'collaborative': collab_scores
        }).fillna(0)
        
        hybrid_df['final_score'] = (hybrid_df['content'] * (w_content / 100)) + (hybrid_df['collaborative'] * (w_collab / 100))
        final_rank = hybrid_df.drop(target_movie, errors='ignore').sort_values(by='final_score', ascending=False)

        st.subheader("🎯 하이브리드 엔진 종합 스코어 랭킹")
        if final_rank.empty:
            st.info("비교 분석할 다른 영화가 영화 보관함에 존재하지 않습니다.")
        else:
            for title, row in final_rank.iterrows():
                target_rows = movies_db[movies_db['title'] == title]
                if target_rows.empty: continue
                target_row = target_rows.iloc[0]
                
                c1, c2 = st.columns([1, 5])
                with c1:
                    if target_row['poster']: st.image(target_row['poster'], width=100)
                with c2:
                    st.write(f"### **{title}** (종합 가중 점수: `{row['final_score']:.2f}`)")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🏷️ 장르 유사 점수", f"{row['genre']:.2f}")
                    m2.metric("📖 줄거리 유사 점수", f"{row['plot']:.2f}")
                    m3.metric("🧬 메타데이터 점수", f"{row['content']:.2f}")
                    m4.metric("👥 인구집단 점수", f"{row['collaborative']:.2f}")
                st.markdown("---")
    except Exception as e:
        st.error(f"연산 중 오류가 발생했습니다: {e}")
