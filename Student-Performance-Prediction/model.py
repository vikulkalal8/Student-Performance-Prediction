"""Reproducible least-squares regression and principal component regression."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
SYN = ['Study_Hours','Attendance','Previous_Marks','Assignment_Marks','Sleep_Hours']
UCI = ['G1','G2','studytime','absences','failures','freetime']
SPECS = {'synthetic': ('student_data.csv', SYN, 'Final_Marks', 100),
         'uci': ('student-mat.csv', UCI, 'G3', 20),
         'uci_early': ('student-mat.csv', UCI[2:], 'G3', 20)}
LABELS = {'Study_Hours':'Study hours','Attendance':'Attendance (%)','Previous_Marks':'Previous marks','Assignment_Marks':'Assignment marks','Sleep_Hours':'Sleep hours','G1':'First-term grade','G2':'Second-term grade','studytime':'Study time category (1–4)','absences':'Absences','failures':'Past failures','freetime':'Free time category (1–5)'}

def available():
    return [key for key, spec in SPECS.items() if (ROOT/'dataset'/spec[0]).exists()]

def load(mode):
    filename, features, target, scale = SPECS[mode]
    df = pd.read_csv(ROOT/'dataset'/filename, sep=';' if mode.startswith('uci') else ',')
    df = df[features+[target]].apply(pd.to_numeric, errors='raise')
    if df.isna().any().any() or not np.isfinite(df.to_numpy()).all():
        raise ValueError('Dataset contains missing or non-finite values.')
    return df, features, target, scale

def pipeline(k=None):
    steps = [StandardScaler()]
    if k is not None: steps.append(PCA(n_components=k, svd_solver='full'))
    return make_pipeline(*steps, LinearRegression())

def metrics(y, pred):
    mse = float(mean_squared_error(y, pred))
    return dict(mae=float(mean_absolute_error(y,pred)), mse=mse, rmse=float(np.sqrt(mse)), r2=float(r2_score(y,pred)))

@lru_cache(maxsize=3)
def experiment(mode='synthetic'):
    df, features, target, scale = load(mode)
    X, y = df[features], df[target]
    train, test, yt, ytest = train_test_split(X,y,test_size=.2,random_state=42)
    p = len(features)
    baseline = pipeline().fit(train,yt)
    scaler = baseline.named_steps['standardscaler']
    z = scaler.transform(train)
    full = PCA(n_components=p,svd_solver='full').fit(z)
    cov = np.cov(z,rowvar=False)
    cv = KFold(n_splits=5,shuffle=True,random_state=42)
    # Choose k using only the training data. Each fold refits its scaler and PCA.
    rows, models, predictions = [], {}, {}
    for k in [None,*range(1,p+1)]:
        model = baseline if k is None else pipeline(k).fit(train,yt)
        pred = model.predict(test)
        score = -cross_val_score(pipeline(k),train,yt,cv=cv,scoring='neg_mean_squared_error')
        key = 'baseline' if k is None else str(k)
        rows.append(dict(key=key, components=p if k is None else k, variance=100.0 if k is None else float(full.explained_variance_ratio_[:k].sum()*100), cv_rmse=float(np.sqrt(score.mean())), **metrics(ytest,pred)))
        models[key] = model
        predictions[key] = pred.tolist()
    best = min(rows[1:],key=lambda row: row['cv_rmse'])['key']
    bounds = {f:dict(label=LABELS[f],min=float(X[f].min()),max=float(X[f].max()),default=float(X[f].median())) for f in features}
    result = dict(mode=mode, rows=len(df), train=len(train), test=len(test), features=features, target=target, scale=scale, bounds=bounds, labels=[LABELS[f] for f in features], comparison=rows, selected=best, explained=(full.explained_variance_ratio_*100).tolist(), eigenvalues=full.explained_variance_.tolist(), eigenvectors=full.components_.T.tolist(), covariance=cov.tolist(), correlation=X.corr().to_numpy().tolist(), actual=ytest.tolist(), predictions=predictions, means=scaler.mean_.tolist(), std=scaler.scale_.tolist(), preview=df.head(15).to_dict('records'), columns=features+[target])
    return result,models

def predict(mode, key, values):
    result, models = experiment(mode)
    if key not in models or set(values)!=set(result['features']): raise ValueError('Invalid model or features.')
    numbers = {}
    for f in result['features']:
        v = float(values[f]); b = result['bounds'][f]
        if not np.isfinite(v) or not b['min'] <= v <= b['max']: raise ValueError(f'{f}: outside dataset range.')
        if mode.startswith('uci') and not v.is_integer(): raise ValueError(f'{f}: use integers.')
        numbers[f] = v
    x = pd.DataFrame([numbers],columns=result['features'])
    model = models[key]; scaler = model.named_steps['standardscaler']; z = scaler.transform(x)
    pcs = model.named_steps['pca'].transform(z)[0].tolist() if key!='baseline' else None
    lr = model.named_steps['linearregression']
    return dict(prediction=float(model.predict(x)[0]), baseline=float(models['baseline'].predict(x)[0]), standardized=z[0].tolist(), pcs=pcs, coefficients=lr.coef_.tolist(), intercept=float(lr.intercept_))
