import pandas as pd
import numpy as np

path = "/home/HTNclinical/Select-Optimal-Decisions-via-DRO-KNN-master/training-data/HTN_RegistryOutput.csv"
df = pd.read_csv(path,nrows=1000,on_bad_lines='skip')

not_use_columns = ["PatientEpicKey","PatientDurableKey","LastServDate_UrineCult",
                       "LastServDate_OfficeVisit","PatientEndDate","LookBackDate","SBPFuture_Avg"]
prescription_columns = [
        "Thiazide90_Ind", "CalciumChannelBlock90_Ind", "ARB90_Ind","ACEI90_Ind", "BetaBlock90_Ind",
        "LoopDiuretic90_Ind", "MineralcorticoidRecAnt90_Ind"
    ]
hist_pres_columns = ["Thiazide_Ind", "CalciumChannelBlock_Ind", "ARB_Ind","ACEI_Ind", "BetaBlock_Ind",
        "LoopDiuretic_Ind", "Leuprolide22_Ind", "Cyclopentolate1Pct_Ind", "Augmentin875_Ind", "MineralcorticoidRecAnt_Ind"
    ]
useful_feature = [item for item in df.columns.tolist()
                      if item not in not_use_columns and item not in prescription_columns and item not in hist_pres_columns]

print(df.shape)
print(df.head())
# Identify categorical columns
potential_categorical_columns = df.select_dtypes(include=['bool', 'object', 'category', 'int64']).columns

# Filter to include only columns with 0 and 1 (considering NaN)
categorical_columns = [col for col in potential_categorical_columns if df[col].dropna().isin([0, 1]).all()]

# Include int columns which could be categorical (in case of int columns with NaNs)
categorical_columns += [col for col in df.select_dtypes(include=['int64']).columns if df[col].dropna().isin([0, 1]).all()]

# Remove duplicates if any
X = pd.get_dummies(df[useful_feature], columns = ["LegalSex","GeneralRace","SmokingStatus"],dtype=int).to_numpy()
cols = [
    "SBP90_Avg", "SBP90180_Avg", "SBP180270_Avg","DBP90_Avg","DBP90180_Avg","DBP180270_Avg", 
    "Temp90_Avg","Temp90180_Avg","Temp180270_Avg","SPO2_90_Avg","SPO2_90180_Avg","SPO2_180270_Avg",
    "Respiration90_Avg","Respiration90180_Avg","Respiration180270_Avg","Pulse90_Avg","Pulse90180_Avg",
    "Pulse180270_Avg","BMI90_Avg","BMI90180_Avg","BMI180270_Avg"
]
# "SerumPotassium_Avg" is missing
cols = [
    "RedCellDist_Avg","MeanCorpHemo_Avg","PartialPressureCO2_Avg","MeanCorpHemoConcentrate_Avg",
    "AnionGapNoPotassium_Avg","NeutrophilCnt_Avg","LymphocytesCnt_Avg","MonocytesPct",
    "Monocytes_Avg","Basophils_Avg","BodySurfaceArea","Eosinophils_Avg",
    "NucleatedRBC_Avg","AbsNeutrophilPCT_Avg","UrinePH_Avg","UrineSpecificGravity_Avg",
    "Glucose_Avg","Leukocyte_Avg","GlucoseWBGV_Avg",
    "EosinophilsPCT_Avg","MeanCorpusHemoRBC_Avg","Chloride_Avg","Cholesterol_Avg",
    "Sodium_Avg","SerumCreatinine_Avg","Albumin_Avg","BasophilsPCT_Avg","HDL_Avg",
    "HemoA1C_Avg","AlkalinePhosphate_Avg","LDLDirect_Avg","MeanCorpuscVol_Avg","LymphocytesPCT_Avg",
    "AspartateAminoTran_Avg","AlanineAminoTran_Avg","Bilirubin_Avg","Triglycerides_Avg",
    "Hematocrit_Avg","Hemoglobin_Avg","Platelet_Avg","BloodUreaNitrogen_Avg"
]
for col in cols:
    if col not in df.columns:
        print(col)
