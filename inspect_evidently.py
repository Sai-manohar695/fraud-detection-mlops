import pandas as pd
import numpy as np
import json
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

df_ref = pd.DataFrame({
    'a': np.random.randn(500),
    'b': np.random.randn(500)
})
df_cur = pd.DataFrame({
    'a': np.random.randn(500) * 5,
    'b': np.random.randn(500) * 5
})

ref_ds = Dataset.from_pandas(df_ref, data_definition=DataDefinition())
cur_ds = Dataset.from_pandas(df_cur, data_definition=DataDefinition())

report   = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

result = snapshot.dump_dict()
print(json.dumps(result, indent=2, default=str)[:3000])