import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics


st.set_page_config(page_title='оценка вина', layout='wide')
st.title('оценка качества вина')


#4.5

app_dir = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(app_dir / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(app_dir / 'test.csv')


@st.cache_data
def get_proba(_model, x):
    return _model.predict_proba(x)[:, 1]


model = load_model()
test = load_test()

config = json.loads(
    (app_dir / 'config.json').read_text(encoding='utf-8')
)

features = list(model.feature_names_in_)
x_test = test[features]
y_test = test['target']

p_test = get_proba(model, x_test)

threshold = st.sidebar.slider(
    'порог',
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.001,
    format='%.3f',
    key='threshold'
)

tab1, tab2, tab3, tab4 = st.tabs([
    'одно вино',
    'модельные метрики',
    'пакетное предсказание',
    'коэф модели'
])


#4.1

with tab1:
    values = {}
    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            if feature == 'is_red':
                values[feature] = int(st.toggle('красное вино'))
                st.caption('выключено - белое вино.')
            else:
                values[feature] = st.number_input(
                    feature.lower(),
                    min_value=float(x_test[feature].min()),
                    max_value=float(x_test[feature].max()),
                    value=float(x_test[feature].median()),
                    step=(
                        0.0001
                        if feature in ['density', 'chlorides']
                        else 0.01
                    ),
                    format='%.5f'
                )

    one_wine = pd.DataFrame([values], columns=features)
    probability = float(get_proba(model, one_wine)[0])

    st.metric(
        'вероятность хорошего вина',
        f'{probability:.1%}'
    )

    st.write(
        'решение:',
        'хорошее вино'
        if probability >= threshold
        else 'не хорошее вино'
    )


#4.2

with tab2:
    pred = (p_test >= threshold).astype(int)

    precision = metrics.precision_score(
        y_test, pred, zero_division=0
    )
    recall = metrics.recall_score(
        y_test, pred, zero_division=0
    )
    f1 = metrics.f1_score(
        y_test, pred, zero_division=0
    )

    c1, c2, c3 = st.columns(3)
    c1.metric('precision', f'{precision:.3f}')
    c2.metric('recall', f'{recall:.3f}')
    c3.metric('f1', f'{f1:.3f}')

    matrix = metrics.confusion_matrix(
        y_test, pred, labels=[0, 1]
    )
    tn, fp, fn, tp = matrix.ravel()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics.ConfusionMatrixDisplay(
        matrix,
        display_labels=['не хорошее', 'хорошее']
    ).plot(
        ax=axes[0],
        cmap='Blues',
        colorbar=False
    )
    axes[0].set_title('матрица ошибок')

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], '--')
    axes[1].scatter(
        fp / (fp + tn),
        recall,
        color='red'
    )
    axes[1].set(
        title='roc-кривая',
        xlabel='fpr',
        ylabel='tpr'
    )

    precision_curve, recall_curve, _ = (
        metrics.precision_recall_curve(y_test, p_test)
    )

    axes[2].plot(recall_curve, precision_curve)
    axes[2].axhline(y_test.mean(), linestyle='--')
    axes[2].scatter(recall, precision, color='red')
    axes[2].set(
        title='pr-кривая',
        xlabel='recall',
        ylabel='precision'
    )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'красные точки показывают текущий порог. '
        'если увеличить порог, подборка станет меньше '
        'и больше хороших вин может быть пропущено. '
        'если уменьшить порог, подборка расширится, '
        'но в неё могут попасть посредственные вина.'
    )


#4.3

with tab3:
    uploaded = st.file_uploader(
        'csv с признаками вин без target',
        type='csv'
    )

    if uploaded is not None:
        try:
            uploaded.seek(0)
            batch = pd.read_csv(uploaded)

            missing = [
                feature
                for feature in features
                if feature not in batch.columns
            ]

            if missing:
                st.error(
                    'не хватает столбцов: '
                    + ', '.join(missing).lower()
                )
            else:
                batch_x = batch[features].apply(pd.to_numeric)

                if not np.isfinite(
                    batch_x.to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        'в признаках есть пропуски '
                        'или бесконечные значения.'
                    )

                probabilities = get_proba(model, batch_x)

                result = batch.copy()
                result['вероятность'] = probabilities
                result['решение'] = (
                    probabilities >= threshold
                ).astype(int)

                st.caption(
                    'решение 1 - хорошее вино, '
                    '0 - не хорошее.'
                )

                st.dataframe(result.rename(columns=str.lower))

                st.download_button(
                    'скачать предсказания',
                    data=result.to_csv(index=False).encode('utf-8-sig'),
                    file_name='wine_predictions.csv',
                    mime='text/csv'
                )

        except ValueError as error:
            st.error(
                f'не удалось обработать csv: {error}'.lower()
            )


#4.4

with tab4:
    coefs = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=[feature.lower() for feature in features]
    ).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))

    coefs.plot.barh(
        ax=ax,
        color=np.where(coefs >= 0, 'green', 'red')
    )

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('коэф после масштабирования')

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    positive = coefs[coefs > 0].nlargest(3).index
    negative = coefs[coefs < 0].nsmallest(3).index

    st.write(
        'сильнее за хорошее вино:',
        ', '.join(positive)
    )

    st.write(
        'сильнее против:',
        ', '.join(negative)
    )

    st.caption(
        'при прочих равных положительный коэф '
        'повышает вероятность хорошего вина, '
        'а отрицательный понижает.'
    )