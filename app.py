import streamlit as st
import pandas as pd
from heavy import HeavyAnalyzer
from light import LightAnalyzer
import config


def main():
    st.set_page_config(page_title="📊 Тематический анализ", layout="wide")
    
    if 'analysis_df' not in st.session_state: st.session_state.analysis_df = None
    if 'selected_topic' not in st.session_state: st.session_state.selected_topic = 0
    
    st.title("📊 Анализатор текста по тематическим блокам")
    st.markdown("""
    Анализирует текст и выделяет **тематические блоки** вместо отдельных предложений.
    """)
    
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        mode = st.radio(
            "Режим анализа:",
            ["🧠 Heavy (LaBSE + AI)", "⚡ Light (Быстрый)"],
            help="Heavy: AI эмоции и пересказ. Light: словарь и ключевые слова."
        )
        
        input_type = st.radio("Способ ввода:", ["📁 Файл (.txt/.md)", "✍️ Текст вручную"])
        
        st.markdown("---")
        
        with st.expander("🛠 Параметры алгоритма", expanded=False):
            st.caption("Тонкая настройка разбиения на темы")
            
            min_sent_len = st.number_input(
                "Мин. длина фразы (симв.)", 
                min_value=5, max_value=100, 
                value=config.MIN_SENTENCE_LENGTH,
                help="Фразы короче этого значения будут отброшены."
            )
            
            if "Heavy" in mode:
                dist_threshold = st.slider(
                    "Порог объединения (Threshold)",
                    min_value=0.1, max_value=1.0, 
                    value=config.HEAVY_DISTANCE_THRESHOLD, step=0.05,
                    help="Меньше = больше мелких тем. Больше = меньше крупных тем."
                )
                light_clusters = None
                st.info(f"Threshold: {dist_threshold}")
            else:
                light_clusters = st.slider(
                    "Количество тем",
                    min_value=2, max_value=20, 
                    value=config.LIGHT_N_CLUSTERS,
                    help="На сколько тем делить текст."
                )
                dist_threshold = None
    
    text = None
    if input_type == "📁 Файл (.txt/.md)":
        uploaded_file = st.file_uploader("Выбери файл:", type=["txt", "md"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.success(f"✅ {uploaded_file.name} загружен")
    else:
        text = st.text_area("Введи текст:", height=200, placeholder="Текст для анализа...")
    
    if st.button("🚀 Анализировать", type="primary", use_container_width=True):
        if not text or len(text.strip()) < 20:
            st.error("❌ Текст слишком короткий")
            return
        
        with st.spinner("🔄 Анализирую..."):
            try:
                is_heavy = "Heavy" in mode
                analyzer = HeavyAnalyzer() if is_heavy else LightAnalyzer()
                
                df = analyzer.analyze(
                    text, 
                    threshold=dist_threshold, 
                    n_clusters=light_clusters,
                    min_length=min_sent_len
                )
                
                if df.empty:
                    st.warning("⚠️ Не удалось выделить темы. Попробуйте изменить параметры.")
                    return
                
                st.session_state.analysis_df = df
                st.session_state.selected_topic = 0
                st.success(f"✅ Найдено {len(df)} тем!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                return
    
    if st.session_state.analysis_df is not None:
        df = st.session_state.analysis_df
        
        st.markdown("---")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📌 Тем", len(df))
        c2.metric("😊 Позитив", len(df[df['sentiment'] == 'positive']))
        c3.metric("😞 Негатив", len(df[df['sentiment'] == 'negative']))
        c4.metric("😐 Нейтраль", len(df[df['sentiment'] == 'neutral']))
        
        st.markdown("---")
        
        st.subheader("📋 Результаты")
        df_display = df.copy()
        df_display['sentiment'] = df_display['sentiment'].map({'positive':'😊 positive', 'negative':'😞 negative', 'neutral':'😐 neutral'})
        cols_hide = ['emotions_json', 'top_emotion']
        st.dataframe(df_display[[c for c in df_display.columns if c not in cols_hide]], use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.subheader("🔍 Детальный просмотр")
        cols = st.columns(len(df))
        for idx, col in enumerate(cols):
            with col:
                row = df.iloc[idx]
                if 'top_emotion' in row and row['top_emotion']:
                    label = f"{row['top_emotion']}\n{row['percentage']}"
                else:
                    icon = {'positive':'😊', 'negative':'😞', 'neutral':'😐'}.get(row['sentiment'], '😐')
                    label = f"{icon} Тема {idx+1}\n{row['percentage']}"
                
                type_btn = "primary" if st.session_state.selected_topic == idx else "secondary"
                if st.button(label, key=f"btn_{idx}", use_container_width=True, type=type_btn):
                    st.session_state.selected_topic = idx
                    st.rerun()
        
        st.markdown("---")
        
        topic = df.iloc[st.session_state.selected_topic]
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown(f"## {topic['theme']}")
            
            if 'emotions_json' in topic and isinstance(topic['emotions_json'], dict) and topic['emotions_json']:
                st.caption("Спектр эмоций:")
                sorted_em = dict(sorted(topic['emotions_json'].items(), key=lambda x: x[1], reverse=True)[:6])
                st.bar_chart(sorted_em, color="#ff4b4b")
            else:
                st.metric("Сентимент", topic['sentiment'])
                
            st.metric("Доля текста", topic['percentage'])
            st.metric("Предложений", topic['n_sentences'])

        with c2:
            st.markdown("### 📝 Краткий пересказ (Summary):")
            st.info(topic['summary'])
            
            st.markdown("### 🔑 Ключевые слова:")
            st.success(topic['keywords'])
            
            st.markdown("### ⭐ Главное предложение:")
            st.warning(topic['key_sentence'])
            
        st.markdown("---")
        
        # ЭКСПОРТ
        st.subheader("💾 Экспорт")
        st.download_button("📥 CSV", df.to_csv(index=False), "analysis.csv", "text/csv")
        st.download_button("📥 JSON", df.to_json(orient='records', force_ascii=False, indent=2), "analysis.json", "application/json")


if __name__ == "__main__":
    main()
