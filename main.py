import os
import sys

def main():
    print("🚀 Запуск дашборда для тематического анализа...")
    try:
        import streamlit
    except ImportError:
        print("Ошибка: Streamlit не установлен. Пожалуйста, установите его:")
        print("pip install streamlit")
        sys.exit(1)
        
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()
