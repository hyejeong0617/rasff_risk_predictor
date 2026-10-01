"""Run: python -m streamlit run app.py"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rasff_core import DATA, ARTIFACTS, load_prepared, predict_bundle

st.set_page_config(page_title='RASFF Explorer & Label Classifier',layout='wide')
st.title('RASFF Explorer & Label Classifier')
st.caption('Historical notified events, not all tested food. Model output estimates a recorded serious label, not recall or actual food safety.')

@st.cache_data
def data(): return load_prepared()

@st.cache_resource
def load_model(stamp): return joblib.load(ARTIFACTS/'final_bundle.joblib')

if not (DATA/'prepared.csv').exists():
    st.info('Run Notebook 01 first to prepare the supplied data.');st.stop()
d=data()
explorer, predictor, evaluation=st.tabs(['Explore notifications','Predict recorded label','Model & evaluation'])
with explorer:
    st.subheader('Filter historical notifications')
    years=st.multiselect('Years',sorted(d.date.dt.year.unique()),default=sorted(d.date.dt.year.unique()))
    dates=st.date_input('Date range',value=(d.date.min().date(),d.date.max().date()),min_value=d.date.min().date(),max_value=d.date.max().date())
    filter_cols=['type','category','notifying_country','classification','risk_decision']
    selections={}
    grid=st.columns(3)
    for i,c in enumerate(filter_cols):
        with grid[i%3]: selections[c]=st.multiselect(c,sorted(d[c].dropna().unique()))
    all_origins=sorted({x.strip() for s in d.origin for x in s.split(',') if x.strip()})
    origins=st.multiselect('Origin countries (any match)',all_origins)
    query=st.text_input('Search subject or original hazards (literal text)')
    f=d[d.date.dt.year.isin(years)].copy()
    if isinstance(dates,(tuple,list)) and len(dates)==2:
        f=f[(f.date.dt.date>=dates[0])&(f.date.dt.date<=dates[1])]
    for c,values in selections.items():
        if values:f=f[f[c].isin(values)]
    if origins:f=f[f.origin.map(lambda s:bool(set(x.strip() for x in s.split(','))&set(origins)))]
    if query:f=f[f.subject.str.contains(query,case=False,regex=False)|f.hazards.str.contains(query,case=False,regex=False)]
    a,b,c=st.columns(3)
    a.metric('Notifications',len(f));b.metric('Recorded serious share',f'{f.target.mean():.1%}' if len(f) else '—')
    c.metric('Missing original hazards',int(f.hazards.eq('').sum()))
    st.caption('Shares describe selected notifications only; no population risk rates or country safety rankings.')
    if not f.empty:
        monthly=f.assign(period=f.date.dt.to_period('M').astype(str)).groupby(['period','risk_decision']).size().reset_index(name='notifications')
        st.plotly_chart(px.bar(monthly,x='period',y='notifications',color='risk_decision',title='Monthly notification counts'),width='stretch')
        dimension=st.selectbox('Compare an item',['category','type','notifying_country','classification','risk_decision','origin'])
        counts=f[dimension].replace('','(missing)').value_counts().head(20).rename_axis(dimension).reset_index(name='notifications')
        st.plotly_chart(px.bar(counts,x='notifications',y=dimension,orientation='h',title='Top 20 recorded values'),width='stretch')
        if dimension=='origin':st.caption('This chart counts complete recorded country combinations, not individual-country risks.')
    visible=['reference','date','subject','category','type','origin','notifying_country','hazards','classification','risk_decision']
    st.dataframe(f[visible],width='stretch',hide_index=True)
    st.download_button('Download filtered notifications',f.to_csv(index=False).encode('utf-8-sig'),'rasff_filtered.csv','text/csv')
with predictor:
    model_path=ARTIFACTS/'final_bundle.joblib'
    if not model_path.exists():st.info('Run Notebooks 02 and 03 to train and save the model.')
    else:
        bundle=load_model(model_path.stat().st_mtime_ns)
        options=json.loads((ARTIFACTS/'input_options.json').read_text())
        st.subheader('Estimate the recorded serious label')
        st.write('Selected candidate:',bundle['spec']['name'])
        st.caption('Inputs match the saved pipeline. Explorer filters do not change or retrain this model.')
        with st.form('prediction'):
            row={}
            for col in bundle['columns']:
                if col=='subject':row[col]=st.text_area('subject — notification text')
                elif col=='hazards':row[col]=st.text_area('hazards — original recorded text (leave blank if missing)')
                elif col=='origin':
                    countries=sorted({x.strip() for s in options[col] for x in s.split(',') if x.strip()})
                    chosen=st.multiselect('origin — choose zero or more countries',countries)
                    row[col]=','.join(chosen)
                else: row[col]=st.selectbox(col,['']+options.get(col,[]))
            submitted=st.form_submit_button('Predict recorded label')
        if submitted:
            if not row['subject'].strip():st.warning('Enter a subject before predicting.')
            else:
                probability=float(predict_bundle(bundle,pd.DataFrame([row]))[0]);positive=probability>=bundle['threshold']
                st.metric('Estimated P(recorded serious)',f'{probability:.1%}')
                st.write('Predicted label:', 'serious' if positive else 'Other recorded labels (not safe)')
                st.caption('Fixed decision threshold: 0.50. A non-serious prediction does not mean safe. Calibration is evaluated on historical held-out data and may not transfer to future records.')
                if any(not str(v).strip() for k,v in row.items() if k not in ['hazards']):
                    st.warning('Some inputs are missing. This output should be interpreted with caution.')
                st.json(row)
with evaluation:
    st.subheader('Model scope and held-out evaluation')
    report_path=ARTIFACTS/'test_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());st.json(report['candidate'])
        st.write('Test dates:',report['test_dates']);st.write('Test rows:',report['test_rows'])
        st.dataframe(pd.DataFrame({k:report[k] for k in ['calibrated','uncalibrated','prior_baseline']}),width='stretch')
        st.dataframe(pd.DataFrame(report['confusion_matrix'],index=['Actual other','Actual serious'],columns=['Predicted other','Predicted serious']))
        st.caption('Candidate selected using forward CV. Calibration is fitted on its own later period; final test is untouched by selection and calibration.')
    else:st.info('Final evaluation is not available yet.')
    st.write('Scope: retrospective classification of existing notifications. Public fields may have been updated after the recorded notification date; no proven pre-decision availability.')
    st.write('Data provenance: inherited historical clean snapshot; raw malformed records are logged separately. Food, feed and other supplied types are included unless filtered in Explorer.')
    st.write('No LLM calls. Existing Hazard_Type and simplified_hazard are optional ablation inputs only.')
