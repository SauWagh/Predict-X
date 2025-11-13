from django.shortcuts import render, redirect
from .models import Document
from .forms import DocumentForm
import pandas as pd
import json
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
import numpy as np
from django.http import HttpResponseRedirect
from django.urls import reverse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from mpl_toolkits.mplot3d import Axes3D

model = None
feature_cols = []
feature_cols_original = []
df_cache = None


def upload_file(request):
    global model, feature_cols, feature_cols_original, df_cache

    df = None
    form = DocumentForm()
    content = None
    doc = None
    prediction_result = None
    manual_form_fields = None
    columns = None

    try:
        if request.method == 'POST':

            if 'file' in request.FILES:
                form = DocumentForm(request.POST, request.FILES)
                if form.is_valid():
                    doc = form.save()
                    filename = doc.file.name.lower()

                    try:
                        if filename.endswith('.csv'):
                            df = pd.read_csv(doc.file)
                        elif filename.endswith(('.xls', '.xlsx')):
                            df = pd.read_excel(doc.file)
                        elif filename.endswith('.json'):
                            data = json.load(doc.file)
                            df = pd.json_normalize(data)
                        elif filename.endswith('.xml'):
                            tree = ET.parse(doc.file)
                            root = tree.getroot()
                            data = [{child.tag: child.text for child in elem} for elem in root]
                            df = pd.DataFrame(data)
                        else:
                            prediction_result = 'Unsupported file type. Please upload CSV, Excel, JSON, or XML.'
                            df = None

                        if df is not None and not df.empty:
                            df = df.dropna(how='all')
                            df_cache = df
                            columns = df.columns.tolist()
                            limited_df = df.head(500)
                            content = limited_df.to_html(classes='table table-striped', index=False)
                            prediction_result = 'Select which column you want to predict.'
                        else:
                            content = None
                            prediction_result = 'Uploaded file is empty or invalid.'

                    except Exception as e:
                        prediction_result = f'Error reading file: {e}'
                        content = None

            elif request.POST.get('select_target'):
                target_col = request.POST.get('target_column')
                if df_cache is not None and target_col:
                    try:
                        df = df_cache.copy()
                        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                        df = df.dropna(subset=[target_col])

                        feature_cols_original = [c for c in df.columns if c != target_col]
                        X = pd.get_dummies(df[feature_cols_original], drop_first=True)
                        y = pd.to_numeric(df[target_col], errors='coerce')

                        if X.empty or y.empty:
                            prediction_result = "Error: Dataset is empty after cleaning."
                        else:
                            model = LinearRegression()
                            model.fit(X, y)
                            feature_cols = X.columns.tolist()
                            manual_form_fields = feature_cols_original
                            prediction_result = f"Model trained to predict '{target_col}' using {len(feature_cols)} features."
                    except Exception as e:
                        prediction_result = f'Error training model: {e}'

            elif request.POST.get('manual_submit'):
                if model is not None and feature_cols:
                    try:
                        user_input = {col: request.POST.get(col, '') for col in feature_cols_original}
                        input_df = pd.DataFrame([user_input])
                        input_encoded = pd.get_dummies(input_df)
                        input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
                        pred = model.predict(input_encoded)
                        prediction_result = f"Predicted value: {round(pred[0], 2)}"
                        return redirect(f"{reverse('results')}?pred={pred[0]}&inputs={','.join(map(str, user_input.values()))}")
                    except Exception as e:
                        prediction_result = f"Error during prediction: {e}"

        else:
            form = DocumentForm()
            content = None

        if df_cache is not None:
            limited_df = df_cache.head(500)
            content = limited_df.to_html(classes='table table-striped', index=False)
            columns = df_cache.columns.tolist()

    except Exception as e:
        prediction_result = f"Unexpected error: {e}"

    return render(request, 'predict_app/index.html', {
        'form': form,
        'doc': doc,
        'content': content,
        'prediction_result': prediction_result,
        'manual_form_fields': manual_form_fields,
        'columns': columns,
    })


def results_view(request):
    global df_cache, model, feature_cols

    if df_cache is None:
        return HttpResponseRedirect(reverse('upload_file'))

    graph_type = request.GET.get('graph_type', 'heatmap')
    prediction_value = request.GET.get('pred', None)
    input_value = request.GET.get('inputs', None)
    images = []

    try:
        df = df_cache.copy()
        plt.switch_backend('Agg')

        if graph_type == 'heatmap':
            plt.figure(figsize=(7, 6))
            corr = df.select_dtypes(include='number').corr()
            plt.imshow(corr, cmap='coolwarm', interpolation='none')
            plt.colorbar()
            plt.title('Correlation Heatmap')
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
            plt.yticks(range(len(corr.columns)), corr.columns)

        elif graph_type == 'scatter' and 'x' in df.columns and 'y' in df.columns:
            plt.figure(figsize=(7, 5))
            plt.scatter(df['x'], df['y'], c='blue', alpha=0.6, label='Data Points')
            if prediction_value:
                plt.scatter(df['x'].iloc[-1], float(prediction_value), c='red', s=100, label='Prediction')
            plt.title('Scatter Plot')
            plt.legend()

        elif graph_type == 'step':
            plt.figure(figsize=(7, 5))
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) >= 1:
                plt.step(range(len(df)), df[numeric_cols[0]], where='mid', label=numeric_cols[0])
                if prediction_value:
                    plt.scatter(len(df), float(prediction_value), c='red', s=80, label='Prediction')
                plt.title('Step Plot')
                plt.legend()

        elif graph_type == 'stackplot':
            plt.figure(figsize=(7, 5))
            numeric_cols = df.select_dtypes(include='number').columns[:3]
            x = np.arange(len(df))
            y = [df[col].values for col in numeric_cols]
            plt.stackplot(x, *y, labels=numeric_cols)
            if prediction_value:
                plt.axhline(float(prediction_value), color='red', linestyle='--', label='Prediction Value')
            plt.legend()
            plt.title('Stack Plot')

        elif graph_type == 'histogram':
            plt.figure(figsize=(7, 5))
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > 0:
                plt.hist(df[numeric_cols[0]], bins=20, color='purple', alpha=0.7)
                if prediction_value:
                    plt.axvline(float(prediction_value), color='red', linestyle='--', linewidth=2,
                                label='Prediction Value')
                plt.title(f'Histogram of {numeric_cols[0]}')
                plt.legend()

        elif graph_type == 'pie':
            plt.figure(figsize=(6, 6))
            if len(df.columns) > 0:
                col = df.select_dtypes(include='object').columns[0] if len(df.select_dtypes(include='object').columns) > 0 else df.columns[0]
                data = df[col].value_counts().head(5)
                plt.pie(data.values, labels=data.index, autopct='%1.1f%%', startangle=90)
                plt.title(f'Pie Chart of {col}')

        elif graph_type == '3d_scatter' and len(df.select_dtypes(include='number').columns) >= 3:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            cols = df.select_dtypes(include='number').columns[:3]
            ax.scatter(df[cols[0]], df[cols[1]], df[cols[2]], c='blue', alpha=0.6)
            if prediction_value:
                ax.scatter(df[cols[0]].iloc[-1], df[cols[1]].iloc[-1],
                           float(prediction_value), c='red', s=100, label='Prediction')
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_zlabel(cols[2])
            plt.title('3D Scatter Plot')

        elif graph_type == '3d_triangular' and len(df.select_dtypes(include='number').columns) >= 3:
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            cols = df.select_dtypes(include='number').columns[:3]
            x, y, z = df[cols[0]], df[cols[1]], df[cols[2]]
            ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
            if prediction_value:
                ax.scatter(x.iloc[-1], y.iloc[-1], float(prediction_value),
                           c='red', s=100, label='Prediction')
            plt.title('3D Triangular Surface')

        elif graph_type == 'prediction' and prediction_value:
            plt.figure(figsize=(5, 4))
            plt.bar(['Predicted Value'], [float(prediction_value)], color='orange')
            plt.title('Prediction Output')

        else:
            plt.text(0.5, 0.5, 'Invalid graph selection or data missing',
                     ha='center', va='center', fontsize=12)

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        images.append(base64.b64encode(buf.getvalue()).decode('utf-8'))

    except Exception as e:
        images = [f"Error creating graph: {e}"]

    return render(request, 'predict_app/results.html', {
        'images': images,
        'predicted_value': prediction_value,
        'input_values': input_value.split(',') if input_value else None,
        'graph_type': graph_type,
    })


def home(request):
    from data_app.models import Dataset
    dataset = Dataset.objects.all()
    return render(request, 'predict_app/home.html', {'dataset': dataset})
