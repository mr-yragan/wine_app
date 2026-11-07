у меня python: 
PS C:\Users\kondr\Desktop\HSE\ММАД\ДЗ\ДЗ_3> python --version
Python 3.12.10

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py

модель загружается через st.cache_resource, тк нужен один общий объект в памяти. таблица test и вероятности сохраняются через st cache_data, тк это данные, для которых streamlit возвращает отдельные копии.

без декоратора joblib.load выполнялся бы при каждом перезапуске скрипта, например, после движения ползунка. вероятности кэшируются отдельно, поэтому при смене порога модель не вызывается повторно - меняется сравнение с порогом и пересчитываются метрики.