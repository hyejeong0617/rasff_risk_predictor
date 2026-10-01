"""Meaningful checks for time separation, unseen data, serialization and UI parity."""
import json
import joblib
import numpy as np
import pandas as pd
from rasff_core import *

def run_checks():
    d=load_prepared();blocks=temporal_blocks(d)
    for a,b in zip(list(blocks)[:-1],list(blocks)[1:]):
        assert blocks[a].date.max()<blocks[b].date.min()
        assert not set(blocks[a].reference)&set(blocks[b].reference)
    # A completely unseen token/category must not be learned from holdout rows.
    train=blocks['train'].head(1200)
    pipe=make_pipeline('lr','raw').fit(train[feature_columns('raw')],train.target)
    future=blocks['test'].head(2).copy()
    future['subject']='holdout_only_unique_token_xyz'
    future['category']='unseen_product_category_xyz'
    p=pipe.predict_proba(future[feature_columns('raw')])[:,1]
    assert np.isfinite(p).all()
    assert 'holdout_only_unique_token_xyz' not in pipe['preprocess'].named_transformers_['subject'].vocabulary_
    encoder=pipe['preprocess'].named_transformers_['categories']
    assert all('unseen_product_category_xyz' not in categories for categories in encoder.categories_)
    # The saved model must use original variables, not derived fields or late actions.
    bundle=joblib.load(ARTIFACTS/'final_bundle.joblib')
    assert not set(bundle['columns'])&set(['Hazard_Type','simplified_hazard']+LATE_COLUMNS)
    sample=blocks['test'].head(10)
    a=predict_bundle(bundle,sample)
    b=predict_bundle(joblib.load(ARTIFACTS/'final_bundle.joblib'),sample)
    np.testing.assert_allclose(a,b)
    # AppTest performs a real prediction and date/year filtering using the exact bundle.
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    assert not at.exception
    at.multiselect[0].set_value([2025]).run()
    count=next(x.value for x in at.metric if x.label=='Notifications')
    assert int(count)==int(d.date.dt.year.eq(2025).sum())
    text='Salmonella in sesame seeds from India'
    at.text_area[0].set_value(text)
    if len(at.text_area)>1:at.text_area[1].set_value('salmonella')
    at.button[0].click().run()
    assert not at.exception
    actual=float(next(x.value for x in at.metric if x.label=='Estimated P(recorded serious)').rstrip('%'))/100
    row={c:'' for c in bundle['columns']};row.update(subject=text,hazards='salmonella')
    expected=float(predict_bundle(bundle,pd.DataFrame([row]))[0])
    assert abs(actual-expected)<.00051
    result={'date_separation':'passed','unknown_token_and_category':'passed',
        'default_feature_scope':'passed','save_reload_parity':'passed','app_filter':'passed','app_prediction_parity':'passed'}
    write_json(ARTIFACTS/'verification.json',result)
    print(result)

if __name__=='__main__':run_checks()
