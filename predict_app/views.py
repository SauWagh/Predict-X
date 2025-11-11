from django.shortcuts import render,redirect
from pyautogui import center
from .models import Document
from .forms import DocumentForm
import pandas as pd
import json
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
import numpy as np
from django.http import HttpResponseRedirect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from django.http import HttpResponseRedirect
from django.urls import reverse
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors


model = None
feature_cols = []
df_cash = None

def upload_file(request):
    global model, feature_cols, df_cash

    df = None
    form = DocumentForm()
    content = None
    doc = None
    prediction_result = None
    manual_form_fields = None
    columns = None

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

                    if df is not None:
                        df = df.dropna()
                        df_cash = df
                        columns = df.columns.tolist()

                        limited_df = df.head(500)
                        content = limited_df.to_html(classes='table table-striped', index=False)
                        prediction_result = 'Select which column you want to predict.'

                except Exception as e:
                    prediction_result = f'Error reading file: {e}'

        elif request.POST.get('select_target'):
            target_col = request.POST.get('target_column')
            if df_cash is not None and target_col:
                try:
                    df = df_cash.copy()
                    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                    df = df.dropna(subset=[target_col])
                    numeric_col = df.select_dtypes(include='number').columns.tolist()
                    feature_cols = [c for c in numeric_col if c != target_col]

                    if len(feature_cols) > 0:
                        x = df[feature_cols]
                        y = df[target_col]

                        model = LinearRegression()
                        model.fit(x, y)

                        manual_form_fields = feature_cols
                        prediction_result = f"Model trained to predict '{target_col}'."
                    else:
                        prediction_result = 'No numeric columns found for training.'

                except Exception as e:
                    prediction_result = f'Error training model: {e}'

        elif request.POST.get('manual_submit'):
            if model is not None and feature_cols:
                try:
                    value = [float(request.POST.get(col, 0)) for col in feature_cols]
                    sample = np.array([value])
                    pred = model.predict(sample)
                    prediction_result = f"Predicted value: {round(pred[0], 2)}"
                    manual_form_fields = feature_cols

                    return redirect(f"{reverse('results')}?pred={pred[0]}&inputs={','.join(map(str, value))}")
                
                except Exception as e:
                    prediction_result = f"Error during prediction: {e}"

    else:
        form = DocumentForm()

    if df_cash is not None:
        limited_df = df_cash.head(500)
        content = limited_df.to_html(classes='table table-striped', index=False)
        columns = df_cash.columns.tolist()

    return render(request, 'predict_app/index.html', {
        'form': form,
        'doc': doc,
        'content': content,
        'prediction_result': prediction_result,
        'manual_form_fields': manual_form_fields,
        'columns': columns,
    })


df_cash = None 

def results_view(request):
    global df_cash, model, feature_cols

    if df_cash is None:
        return HttpResponseRedirect(reverse('upload_file'))

    graph_type = request.GET.get('graph_type', 'heatmap')
    prediction_value = request.GET.get('pred', None)
    input_value = request.GET.get('inputs', None)
    images = []

    try:
        df = df_cash.copy()
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
            numeric_cols = df.select_dtypes(include='number').columns

            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                data = df[col].value_counts().head(5)

                labels = list(data.index)
                values = list(data.values)

                if prediction_value:
                    try:
                        pred_val = float(prediction_value)
                        labels.append("Predicted Value")
                        values.append(pred_val)
                    except ValueError:
                        pass

                base_colors = list(plt.cm.Paired.colors)
                colors = base_colors[:len(values) - 1] + [(1, 0.2, 0.2, 0.8)] if prediction_value else base_colors[:len(values)]

                wedges, texts, autotexts = plt.pie(
                    values,
                    labels=[f"{l} ({v:.2f})" for l, v in zip(labels, values)],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors[:len(values)],
                    pctdistance=0.8
                )

                plt.setp(autotexts, size=9, color="white", weight="bold")
                plt.title(f'Pie Chart of {col} (Predicted Value Highlighted)')
            else:
                plt.text(0.5, 0.5, "No numeric column available for pie chart",
                 ha='center', va='center', fontsize=12)


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
    return render(request,  'predict_app/home.html',{'dataset':dataset})

