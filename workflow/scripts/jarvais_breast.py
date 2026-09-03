
from damply import dirs
import pandas as pd

from jarvais.analyzer import Analyzer
from jarvais.trainer import TrainerSupervised
from jarvais.explainer import Explainer

data_dir = dirs.PROCDATA / "Breast" / "TCIA_ISPY2" / "BRADCURE_analysis"
output_dir = dirs.RESULTS / "TCIA_ISPY2" / "jarvais"

if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)


clinical_ISPY2 = pd.read_csv(data_dir / "clinical_ISPY2_combined.csv")
clinical_ISPY2 = clinical_ISPY2.drop(['ethnicity', 'Race', 'MP', 'Arm'], axis=1)
# drop HR, PCR, HER2 for HR or HER2 
# clinical_ISPY2 = clinical_ISPY2.drop(['HR', 'pCR'], axis=1)

radiomics_ISPY2 = pd.read_csv(data_dir / "features_only_ISPY2_combined.csv")

data_ISPY2 = pd.merge(clinical_ISPY2, radiomics_ISPY2, left_on="Patient_ID", right_on="Patient_ID")
# data_ISPY2 = clinical_ISPY2
data_ISPY2 = pd.merge(clinical_ISPY2.loc[:, ['Patient_ID', 'HR']], radiomics_ISPY2, left_on="Patient_ID", right_on="Patient_ID")
# data_ISPY2 = pd.merge(clinical_ISPY2, radiomics_ISPY2.loc[:, ['Patient_ID', 'original_shape_MeshVolume']], left_on="Patient_ID", right_on="Patient_ID")
data_ISPY2 = data_ISPY2.drop('Patient_ID', axis=1)

target = 'HR'
feature_num = 50
reduction_method = 'mrmr'
output_dir = output_dir / f"radiomic_{feature_num}" / target

print(f"Initializing Analyzer for {target}...")
# Run Analyzer
analyzer = Analyzer(
    data_ISPY2,
    output_dir = output_dir / f"analyzer_{target}",
    categorical_columns = ['HER2', 'HR', 'pCR' 'menopausal_status'],
    target_variable = target,
    task="classification"
)

# Drop multiplotting, expensive operation
analyzer.settings.visualization.plots.remove('multiplot')

# print(f"Running Analyzer for {target}...")
# analyzer.run()


print(f"Initializing Trainer for {target}...")
# Run Trainer
trainer = TrainerSupervised(
    output_dir= output_dir / f"trainer_{target}",
    target_variable = target,
    task = 'binary',
    k_folds=5,
    reduction_method=reduction_method,
    keep_k=feature_num,
    explain = True
)

print(trainer)
analyzer.data[target] = analyzer.data[target].astype(int)

print(f"Running Trainer for {target}...")
trainer.run(analyzer.data)


# from jarvais.utils.plot import plot_roc_curve, plot_precision_recall_curve

# chosen_model = "LightGBMLarge"
# fig_out_dir = output_dir / f"explainer_{target}/{chosen_model}_figures"
# fig_out_dir.mkdir(parents=True, exist_ok=True)

# pred_train = trainer.infer(trainer.X_train, model=chosen_model)
# pred_val = trainer.infer(trainer.X_val, model=chosen_model)
# pred_test = trainer.infer(trainer.X_test, model = chosen_model)

# plot_roc_curve(trainer.y_test.to_numpy(), pred_test, fig_out_dir,
#                trainer.y_val.to_numpy(), pred_val,
#                trainer.y_train.to_numpy(), pred_train
#                )

# plot_precision_recall_curve(trainer.y_test.to_numpy(), pred_test, fig_out_dir,
#                             trainer.y_val.to_numpy(), pred_val,
#                             trainer.y_train.to_numpy(), pred_train)

