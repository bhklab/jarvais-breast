
from damply import dirs
import pandas as pd

from jarvais.analyzer import Analyzer
from jarvais.trainer import TrainerSupervised


data_dir = dirs.PROCDATA / "Breast" / "TCIA_ISPY2" / "BRADCURE_analysis"
output_dir = dirs.RESULTS / "TCIA_ISPY2" / "jarvais"

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)


clinical_ISPY2 = pd.read_csv(data_dir / "clinical_ISPY2_combined.csv")
clinical_ISPY2 = clinical_ISPY2.drop(['ethnicity', 'Race'], axis=1)

radiomics_ISPY2 = pd.read_csv(data_dir / "features_only_ISPY2_combined.csv")

data_ISPY2 = pd.merge(clinical_ISPY2, radiomics_ISPY2, left_on="Patient_ID", right_on="Patient_ID")
data_ISPY2 = data_ISPY2.drop('Patient_ID', axis=1)



# Run Analyzer
analyzer = Analyzer(
    data_ISPY2,
    output_dir = output_dir / "analyzer_outputs",
    categorical_columns = ['Arm', 'HR', 'HER2', 'MP', 'pCR', 'menopausal_status'],
    target_variable = 'pCR',
    task="classification"
)

# Drop multiplotting, expensive operation
analyzer.settings.visualization.plots.remove('multiplot')

# analyzer.run()


# Run Trainer
trainer = TrainerSupervised(
    output_dir= output_dir / "trainer",
    target_variable = 'pCR',
    task = 'binary',
    k_folds=5,
    reduction_method='mrmr',
    keep_k=50,
    explain=True
)

# print(trainer)
analyzer.data['pCR'] = analyzer.data['pCR'].astype(int)

trainer.run(analyzer.data)