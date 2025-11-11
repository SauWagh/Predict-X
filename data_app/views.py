from django.shortcuts import render, redirect, get_object_or_404
from .models import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from django.shortcuts import render, get_object_or_404
import pandas as pd
import numpy as np
import io, base64
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def home(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv = request.FILES['csv_file']
        dataset = Dataset.objects.create(name=csv.name, file=csv)
        return redirect('dataset_list')
    return render(request, 'data_app/home.html')


def dataset_list(request):
    datasets = Dataset.objects.all().order_by('-uploaded_at')
    return render(request, 'data_app/dataset_list.html', {'datasets': datasets})




def dataset_detail(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    df = pd.read_csv(dataset.file.path)

    columns = df.columns.tolist()
    graphs = []

    # Generate sample graphs for each column
    for col in df.columns[:5]:  # limit to 5 for performance
        plt.figure(figsize=(4, 3))
        if df[col].dtype == 'object':
            df[col].value_counts().plot(kind='bar', title=f"{col} distribution")
        else:
            sns.histplot(df[col], kde=True)
            plt.title(f"{col} distribution")
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs.append(image_base64)
        plt.close()

    # Preview first 500 rows
    data_preview = df.head(500).to_html(classes="table table-striped", index=False)

    return render(request, 'data_app/dataset_detail.html', {
        'dataset': dataset,
        'columns': columns,
        'graphs': graphs,
        'data_preview': data_preview
    })


def prediction(request, pk):
    dataset = get_object_or_404(Dataset, pk=pk)
    df = pd.read_csv(dataset.file.path)

    preview_data = df.head(500).to_html(classes='table table-striped', index=False)
    label_encoders = {}

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    columns = df.columns.tolist()
    result = None
    selected_col = None

    if request.method == 'POST':
        target_col = request.POST['target']
        selected_col = target_col
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=2000) if y.nunique() <= 10 else LinearRegression()
        model.fit(X_train, y_train)

        # Prepare input
        custom_values = []
        for col in X.columns:
            input_val = request.POST.get(col, "")
            if col in label_encoders:
                le = label_encoders[col]
                if input_val not in le.classes_:
                    le.classes_ = np.append(le.classes_, input_val)
                val = le.transform([input_val])[0]
            else:
                try:
                    val = float(input_val)
                except:
                    val = 0
            custom_values.append(val)

        arr = np.array(custom_values).reshape(1, -1)
        prediction_value = model.predict(arr)[0]
        result = f"Predicted value for '{target_col}': {prediction_value:.3f}"

    return render(request, 'data_app/prediction.html', {
        'dataset': dataset,
        'columns': columns,
        'preview_data': preview_data,
        'result': result,
        'selected_col': selected_col
    })
