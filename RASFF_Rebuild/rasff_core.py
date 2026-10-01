"""Reproducible retrospective RASFF classification; no external API calls."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import re
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
LABELS = ['serious', 'not serious', 'undecided', 'potential risk', 'potentially serious', 'no risk']
SAFE_CATS = ['category', 'type', 'notifying_country']
LATE_COLUMNS = ['classification', 'distribution', 'forAttention', 'forFollowUp', 'operator']

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def normalize_frame(frame):
    """Deterministic formatting only; never fits on labels or population statistics."""
    d = frame.copy()
    for c in ['reference', 'subject', 'hazards', 'origin'] + SAFE_CATS + LATE_COLUMNS + ['Hazard_Type', 'simplified_hazard']:
        if c in d:
            d[c] = d[c].astype('string').fillna('').str.strip()
    for c in SAFE_CATS + ['origin', 'classification']:
        if c in d:
            d[c] = d[c].str.lower().str.replace(r'\s+', ' ', regex=True)
    if 'date' in d:
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
    if 'risk_decision' in d:
        d['risk_decision'] = d['risk_decision'].astype('string').str.lower().str.strip()
    return d

def prepare_data(clean_path=DATA / 'CleanedDataset.csv', raw_path=DATA / 'RASFF_originaldataset.csv'):
    """Uses the supplied historical clean snapshot, retaining provenance limitations."""
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    d = normalize_frame(pd.read_csv(clean_path, dtype={'reference': 'string'}))
    required = ['reference', 'date', 'risk_decision', 'subject', 'hazards', 'origin'] + SAFE_CATS
    missing = set(required) - set(d.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    invalid = d['date'].isna() | ~d['risk_decision'].isin(LABELS) | d['reference'].eq('')
    rejected = d.loc[invalid].copy()
    valid = d.loc[~invalid].copy()
    duplicate = valid['reference'].duplicated(keep=False)
    if duplicate.any():
        valid.loc[duplicate].to_csv(ARTIFACTS / 'duplicate_references.csv', index=False)
        raise ValueError('Duplicate references: resolve explicitly before training.')
    valid['target'] = valid['risk_decision'].eq('serious').astype(int)
    valid = valid.sort_values(['date', 'reference']).reset_index(drop=True)
    rejected.to_csv(ARTIFACTS / 'rejected_clean_rows.csv', index=False)
    raw_audit = []
    if Path(raw_path).exists():
        with open(raw_path, encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f); header = next(reader)
            for row in reader:
                if len(row) != len(header):
                    raw_audit.append({'ending_line': reader.line_num, 'field_count': len(row), 'expected_fields': len(header), 'raw_values': json.dumps(row, ensure_ascii=False)})
        pd.DataFrame(raw_audit, columns=['ending_line','field_count','expected_fields','raw_values']).to_csv(ARTIFACTS / 'raw_malformed_rows.csv', index=False)
    report = {'source': str(clean_path), 'source_sha256': hashlib.sha256(Path(clean_path).read_bytes()).hexdigest(),
              'loaded_rows': len(d), 'retained_rows': len(valid), 'rejected_rows': len(rejected),
              'raw_malformed_records': len(raw_audit), 'first_date': valid.date.min(), 'last_date': valid.date.max(),
              'missing_hazards': int(valid.hazards.eq('').sum()), 'label_counts': valid.risk_decision.value_counts().to_dict(),
              'scope': 'All supplied product types, including food, feed and food contact material.',
              'provenance_limit': 'Historical clean snapshot inherits prior edits and exclusions. Raw malformed rows are audited, not silently repaired. Original manual correction (reference 2021.2621) is inherited; source verification remains a follow-up.'}
    write_json(ARTIFACTS / 'data_audit.json', report)
    valid.to_csv(DATA / 'prepared.csv', index=False)
    return valid, report

def load_prepared():
    return normalize_frame(pd.read_csv(DATA / 'prepared.csv', dtype={'reference': 'string'})).assign(
        target=lambda d: d['risk_decision'].eq('serious').astype(int))

def temporal_blocks(d):
    """60/15/10/15 percent of unique observed calendar days, not row counts."""
    days = pd.DatetimeIndex(d.date.dt.normalize().unique()).sort_values()
    if len(days) < 30: raise ValueError('At least 30 distinct dates required.')
    cut = [days[int(len(days)*q)] for q in [.60, .75, .85]]
    day = d.date.dt.normalize()
    out = {'train': d.loc[day < cut[0]].copy(),
           'validation': d.loc[(day >= cut[0]) & (day < cut[1])].copy(),
           'calibration': d.loc[(day >= cut[1]) & (day < cut[2])].copy(),
           'test': d.loc[day >= cut[2]].copy()}
    for name, frame in out.items():
        if frame.target.nunique() != 2: raise ValueError(f'{name} must contain both classes.')
    assert sum(map(len, out.values())) == len(d)
    write_json(ARTIFACTS / 'split_manifest.json', {k:{'rows':len(v),'first':v.date.min(),'last':v.date.max(),'serious_share':float(v.target.mean())} for k,v in out.items()})
    return out

def forward_folds(d, n=3):
    days = pd.DatetimeIndex(d.date.dt.normalize().unique()).sort_values()
    segments = np.array_split(days, n+1)
    for i in range(1, n+1):
        tr = np.flatnonzero(d.date.dt.normalize().isin(np.concatenate(segments[:i])))
        va = np.flatnonzero(d.date.dt.normalize().isin(segments[i]))
        if d.iloc[tr].target.nunique()!=2 or d.iloc[va].target.nunique()!=2:
            raise ValueError('A CV fold has a single class.')
        assert d.iloc[tr].date.max() < d.iloc[va].date.min()
        yield tr, va

def origin_text(series):
    """Country sets become token counts; whitespace normalized without label learning."""
    return series.fillna('').astype(str).map(lambda s:' '.join('country_' + re.sub(r'\W+', '_', x.strip().lower()).strip('_') for x in s.split(',') if x.strip()))

def feature_columns(feature_set='raw', include_classification=False):
    cols = ['subject', 'origin'] + SAFE_CATS
    if feature_set in ['raw', 'derived']: cols += ['hazards']
    if feature_set == 'derived': cols += ['Hazard_Type', 'simplified_hazard']
    if include_classification: cols += ['classification']
    return cols

def make_pipeline(algorithm='lr', feature_set='raw', include_classification=False, c=1.0):
    """Everything learned from corpus data is INSIDE the fitted pipeline."""
    cats = SAFE_CATS.copy()
    if feature_set == 'derived': cats += ['Hazard_Type', 'simplified_hazard']
    if include_classification: cats += ['classification']
    transforms = [
        ('subject', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=6000, sublinear_tf=True), 'subject'),
        ('origin', Pipeline([('tokens', FunctionTransformer(origin_text, validate=False)),
                             ('vector', TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', binary=True, use_idf=False, norm=None))]), 'origin'),
        ('categories', OneHotEncoder(handle_unknown='ignore', min_frequency=10), cats)]
    if feature_set in ['raw','derived']:
        # A constant token prevents an empty vocabulary when every hazard is missing.
        transforms.append(('hazards', Pipeline([('missing', FunctionTransformer(hazard_text, validate=False)),
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=2000, sublinear_tf=True))]), 'hazards'))
    prep = ColumnTransformer(transforms, remainder='drop', sparse_threshold=1.0)
    if algorithm == 'lr': model = LogisticRegression(C=c, max_iter=500, solver='liblinear', random_state=42)
    elif algorithm == 'nb': model = ComplementNB(alpha=1.0)
    elif algorithm == 'xgb':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.05, subsample=.8,
            colsample_bytree=.8, reg_lambda=2., tree_method='hist', n_jobs=2, random_state=42, eval_metric='logloss')
    else: raise ValueError(algorithm)
    return Pipeline([('preprocess', prep), ('model', model)])

def hazard_text(series):
    return series.fillna('').astype(str).replace('', 'missing_hazard')

def metrics(y, p, threshold=.5):
    p = np.clip(np.asarray(p), 1e-7, 1-1e-7); pred = (p>=threshold).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro')),
        'serious_precision':float(precision_score(y,pred,zero_division=0)), 'serious_recall':float(recall_score(y,pred,zero_division=0)),
        'average_precision':float(average_precision_score(y,p)), 'roc_auc':float(roc_auc_score(y,p)),
        'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p)), 'threshold':threshold}

def candidate_specs(include_derived=False, include_late=False):
    specs = [{'name':'core_lr','algorithm':'lr','feature_set':'core','c':1.},
             {'name':'raw_lr_C05','algorithm':'lr','feature_set':'raw','c':.5},
             {'name':'raw_lr_C2','algorithm':'lr','feature_set':'raw','c':2.},
             {'name':'raw_nb','algorithm':'nb','feature_set':'raw','c':1.},
             {'name':'raw_xgb','algorithm':'xgb','feature_set':'raw','c':1.}]
    if include_derived:
        specs.extend([{'name':'ablation_raw_lr','algorithm':'lr','feature_set':'raw','c':1.},
                      {'name':'ablation_derived_lr','algorithm':'lr','feature_set':'derived','c':1.}])
    if include_late: specs.append({'name':'late_classification_lr','algorithm':'lr','feature_set':'raw','c':1.,'include_classification':True})
    return specs

def select_model(d, include_derived=False, include_late=False):
    blocks = temporal_blocks(d); train = blocks['train']; validation = blocks['validation']
    records = []; validations = []; specs = candidate_specs(include_derived, include_late)
    for spec in specs:
        args = {k:v for k,v in spec.items() if k!='name'}; cols = feature_columns(spec['feature_set'],spec.get('include_classification',False))
        if not set(cols).issubset(d.columns): raise ValueError(f'Missing columns for {spec["name"]}')
        for fold,(tr,va) in enumerate(forward_folds(train),1):
            pipe = make_pipeline(**args)
            pipe.fit(train.iloc[tr][cols], train.iloc[tr].target)
            p = pipe.predict_proba(train.iloc[va][cols])[:,1]
            records.append({'candidate':spec['name'],'fold':fold,**metrics(train.iloc[va].target,p)})
        pipe = make_pipeline(**args).fit(train[cols],train.target)
        p = pipe.predict_proba(validation[cols])[:,1]
        validations.append({'candidate':spec['name'],**metrics(validation.target,p)})
        print('Completed:',spec['name'],flush=True)
    cv = pd.DataFrame(records); val = pd.DataFrame(validations)
    # Primary candidate selection: mean forward-CV macro F1. Validation is a locked diagnostic.
    # Late classification scenario NEVER becomes the primary dashboard model.
    summary = cv.groupby('candidate').agg(cv_macro_f1=('macro_f1','mean'),cv_macro_f1_std=('macro_f1','std'),cv_AP=('average_precision','mean')).reset_index()
    primary = summary[~summary.candidate.str.startswith(('late_','ablation_'))].sort_values(['cv_macro_f1','cv_AP'],ascending=False)
    chosen = next(s for s in specs if s['name']==primary.iloc[0].candidate)
    write_json(ARTIFACTS/'selected_spec.json',chosen)
    cv.to_csv(ARTIFACTS/'cv_results.csv',index=False);val.to_csv(ARTIFACTS/'validation_results.csv',index=False)
    summary.to_csv(ARTIFACTS/'candidate_summary.csv',index=False)
    return summary.sort_values('cv_macro_f1',ascending=False), val, chosen

def logit_feature(p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def predict_bundle(bundle, frame):
    x = normalize_frame(frame)
    missing = set(bundle['columns'])-set(x.columns)
    if missing: raise ValueError(f'Required input fields missing: {sorted(missing)}')
    p = bundle['pipeline'].predict_proba(x[bundle['columns']])[:,1]
    return bundle['calibrator'].predict_proba(logit_feature(p))[:,1]

def fit_final(d, force=False):
    """Calibrates on a separate later block; evaluates once on the last block."""
    report_path=ARTIFACTS/'test_report.json'
    if report_path.exists() and not force:
        raise RuntimeError('A test report already exists. Read it instead of retuning on test results. Use a new experiment folder for a new development cycle.')
    spec=json.loads((ARTIFACTS/'selected_spec.json').read_text()); blocks=temporal_blocks(d)
    dev=pd.concat([blocks['train'],blocks['validation']]).sort_values('date')
    cols=feature_columns(spec['feature_set'],spec.get('include_classification',False))
    pipe=make_pipeline(**{k:v for k,v in spec.items() if k!='name'}).fit(dev[cols],dev.target)
    cal=blocks['calibration']; cal_p=pipe.predict_proba(cal[cols])[:,1]
    calibrator=LogisticRegression(C=1.,solver='lbfgs').fit(logit_feature(cal_p),cal.target)
    bundle={'pipeline':pipe,'calibrator':calibrator,'columns':cols,'spec':spec,'threshold':.5,
        'target_definition':'1 = recorded serious; 0 = all other known recorded labels (not safe).',
        'scope':'Retrospective classification of notified events. No recall, contamination or general food safety probability.',
        'sklearn_version':sklearn.__version__,'python_version':platform.python_version(),
        'source_sha256':json.loads((ARTIFACTS/'data_audit.json').read_text())['source_sha256']}
    test=blocks['test']; p=predict_bundle(bundle,test); raw=pipe.predict_proba(test[cols])[:,1]
    dummy=DummyClassifier(strategy='prior').fit(np.zeros((len(dev),1)),dev.target)
    prior=dummy.predict_proba(np.zeros((len(test),1)))[:,1]
    report={'candidate':spec,'source_sha256':bundle['source_sha256'],'test_rows':len(test),'test_dates':[str(test.date.min()),str(test.date.max())],
        'calibrated':metrics(test.target,p),'uncalibrated':metrics(test.target,raw),'prior_baseline':metrics(test.target,prior),
        'development_train':metrics(dev.target,pipe.predict_proba(dev[cols])[:,1]),
        'confusion_matrix_labels':[0,1],'confusion_matrix':confusion_matrix(test.target,(p>=.5).astype(int),labels=[0,1]).tolist(),
        'probability_note':'Sigmoid calibration uses the calibration block only. Assess Brier/log loss and reliability plot; calibration is not a guarantee under future shift.'}
    joblib.dump(bundle,ARTIFACTS/'final_bundle.joblib')
    write_json(report_path,report)
    options={c:sorted(dev[c].dropna().astype(str).unique().tolist()) for c in cols if c not in ['subject','hazards']}
    # App options are derived from development data only, never test data.
    write_json(ARTIFACTS/'input_options.json',options)
    preds=test[['reference','date','subject','risk_decision','target']].copy()
    preds['p_serious']=p;preds['prediction']=(p>=.5).astype(int)
    preds.to_csv(ARTIFACTS/'test_predictions.csv',index=False)
    reloaded=joblib.load(ARTIFACTS/'final_bundle.joblib')
    assert np.allclose(p[:20],predict_bundle(reloaded,test.head(20)))
    return bundle,report
