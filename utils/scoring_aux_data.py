"""
from 3 files of the official evaluation repo:

    dx_mapping_scored.csv, dx_mapping_unscored.csv, weights.csv
"""
from io import StringIO
from typing import Union, Optional, List, Tuple, Sequence, Dict
from numbers import Real

import numpy as np
import pandas as pd
from easydict import EasyDict as ED


__all__ = [
    "df_weights",
    "df_weights_expanded",
    "df_weights_abbr",
    "df_weights_fullname",
    "dx_mapping_scored",
    "dx_mapping_unscored",
    "dx_mapping_all",
    "equiv_class_dict",
    "load_weights",
    "get_class",
    "get_class_count",
    "get_class_weight",
    "normalize_class",
    "dx_cooccurrence_all",
    "dx_cooccurrence_scored",
    "get_cooccurrence",
]

# constants

df_weights = pd.read_csv(StringIO(""",164889003,164890007,6374002,426627000,733534002|164909002,713427006|59118001,270492004,713426002,39732003,445118002,164947007,251146004,111975006,698252002,426783006,284470004|63593006,10370003,365413008,427172004|17338001,164917005,47665007,427393009,426177001,427084000,164934002,59931005
164889003,1.0,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,0.5
164890007,0.5,1.0,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,0.5
6374002,0.475,0.475,1.0,0.325,0.475,0.425,0.325,0.325,0.375,0.375,0.325,0.45,0.475,0.375,0.275,0.3625,0.4,0.45,0.4,0.375,0.375,0.325,0.325,0.4,0.475,0.475
426627000,0.3,0.3,0.325,1.0,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
733534002|164909002,0.475,0.475,0.475,0.325,1.0,0.425,0.325,0.325,0.375,0.375,0.325,0.45,0.475,0.375,0.275,0.3625,0.4,0.45,0.4,0.375,0.375,0.325,0.325,0.4,0.475,0.475
713427006|59118001,0.4,0.4,0.425,0.4,0.425,1.0,0.4,0.4,0.45,0.45,0.4,0.475,0.45,0.45,0.35,0.4375,0.475,0.475,0.475,0.3,0.45,0.4,0.4,0.475,0.4,0.4
270492004,0.3,0.3,0.325,0.5,0.325,0.4,1.0,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
713426002,0.3,0.3,0.325,0.5,0.325,0.4,0.5,1.0,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
39732003,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,1.0,0.5,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
445118002,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,1.0,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
164947007,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,1.0,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,0.5,0.425,0.3,0.3
251146004,0.425,0.425,0.45,0.375,0.45,0.475,0.375,0.375,0.425,0.425,0.375,1.0,0.475,0.425,0.325,0.4125,0.45,0.475,0.45,0.325,0.425,0.375,0.375,0.45,0.425,0.425
111975006,0.45,0.45,0.475,0.35,0.475,0.45,0.35,0.35,0.4,0.4,0.35,0.475,1.0,0.4,0.3,0.3875,0.425,0.475,0.425,0.35,0.4,0.35,0.35,0.425,0.45,0.45
698252002,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,0.5,0.45,0.425,0.4,1.0,0.4,0.4875,0.475,0.425,0.475,0.25,0.5,0.45,0.45,0.475,0.35,0.35
426783006,0.25,0.25,0.275,0.45,0.275,0.35,0.45,0.45,0.4,0.4,0.45,0.325,0.3,0.4,1.0,0.4125,0.375,0.325,0.375,0.15,0.4,0.45,0.45,0.375,0.25,0.25
284470004|63593006,0.3375,0.3375,0.3625,0.4625,0.3625,0.4375,0.4625,0.4625,0.4875,0.4875,0.4625,0.4125,0.3875,0.4875,0.4125,1.0,0.4625,0.4125,0.4625,0.2375,0.4875,0.4625,0.4625,0.4625,0.3375,0.3375
10370003,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,1.0,0.45,0.5,0.275,0.475,0.425,0.425,0.5,0.375,0.375
365413008,0.425,0.425,0.45,0.375,0.45,0.475,0.375,0.375,0.425,0.425,0.375,0.475,0.475,0.425,0.325,0.4125,0.45,1.0,0.45,0.325,0.425,0.375,0.375,0.45,0.425,0.425
427172004|17338001,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,0.5,0.45,1.0,0.275,0.475,0.425,0.425,0.5,0.375,0.375
164917005,0.4,0.4,0.375,0.2,0.375,0.3,0.2,0.2,0.25,0.25,0.2,0.325,0.35,0.25,0.15,0.2375,0.275,0.325,0.275,1.0,0.25,0.2,0.2,0.275,0.4,0.4
47665007,0.35,0.35,0.375,0.45,0.375,0.45,0.45,0.45,0.5,0.5,0.45,0.425,0.4,0.5,0.4,0.4875,0.475,0.425,0.475,0.25,1.0,0.45,0.45,0.475,0.35,0.35
427393009,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,1.0,0.5,0.425,0.3,0.3
426177001,0.3,0.3,0.325,0.5,0.325,0.4,0.5,0.5,0.45,0.45,0.5,0.375,0.35,0.45,0.45,0.4625,0.425,0.375,0.425,0.2,0.45,0.5,1.0,0.425,0.3,0.3
427084000,0.375,0.375,0.4,0.425,0.4,0.475,0.425,0.425,0.475,0.475,0.425,0.45,0.425,0.475,0.375,0.4625,0.5,0.45,0.5,0.275,0.475,0.425,0.425,1.0,0.375,0.375
164934002,0.5,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,1.0,0.5
59931005,0.5,0.5,0.475,0.3,0.475,0.4,0.3,0.3,0.35,0.35,0.3,0.425,0.45,0.35,0.25,0.3375,0.375,0.425,0.375,0.4,0.35,0.3,0.3,0.375,0.5,1.0"""), index_col=0)
df_weights.index = df_weights.index.map(str)


def expand_equiv_classes(df:pd.DataFrame, sep:str="|") -> pd.DataFrame:
    """ finished, checked,

    expand df so that rows/cols with equivalent classes indicated by `sep` are separated

    Parameters:
    -----------
    df: DataFrame,
        the dataframe to be split
    sep: str, default "|",
        separator of equivalent classes

    Returns:
    --------
    df_out: DataFrame,
        the expanded DataFrame
    """
    # check whether df is symmetric
    if not (df.columns == df.index).all() or not (df.values.T == df.values).all():
        raise ValueError("the input DataFrame (matrix) is not symmetric")
    df_out = df.copy()
    col_row = df_out.columns.tolist()
    # df_sep = "\|" if sep == "|" else sep
    new_cols = []
    for c in col_row:
        for new_c in c.split(sep)[1:]:
            new_cols.append(new_c)
            df_out[new_c] = df_out[c].values
            new_r = new_c
            df_out.loc[new_r] = df_out.loc[df_out.index.str.contains(new_r)].values[0]
    col_row = [c.split(sep)[0] for c in col_row] + new_cols
    df_out.columns = col_row
    df_out.index = col_row
    return df_out


df_weights_expanded = expand_equiv_classes(df_weights)
    

dx_mapping_scored = pd.read_csv(StringIO("""Dx,SNOMED CT Code,Abbreviation,CPSC,CPSC-Extra,StPetersburg,PTB,PTB-XL,Georgia,Total,Notes
1st degree av block,270492004,IAVB,722,106,0,0,797,769,2394,
atrial fibrillation,164889003,AF,1221,153,2,15,1514,570,3475,
atrial flutter,164890007,AFL,0,54,0,1,73,186,314,
bradycardia,426627000,Brady,0,271,11,0,0,6,288,
complete right bundle branch block,713427006,CRBBB,0,113,0,0,542,28,683,We score 713427006 and 59118001 as the same diagnosis.
incomplete right bundle branch block,713426002,IRBBB,0,86,0,0,1118,407,1611,
left anterior fascicular block,445118002,LAnFB,0,0,0,0,1626,180,1806,
left axis deviation,39732003,LAD,0,0,0,0,5146,940,6086,
left bundle branch block,164909002,LBBB,236,38,0,0,536,231,1041,
low qrs voltages,251146004,LQRSV,0,0,0,0,182,374,556,
nonspecific intraventricular conduction disorder,698252002,NSIVCB,0,4,1,0,789,203,997,
pacing rhythm,10370003,PR,0,3,0,0,296,0,299,
premature atrial contraction,284470004,PAC,616,73,3,0,398,639,1729,We score 284470004 and 63593006 as the same diagnosis.
premature ventricular contractions,427172004,PVC,0,188,0,0,0,0,188,We score 427172004 and 17338001 as the same diagnosis.
prolonged pr interval,164947007,LPR,0,0,0,0,340,0,340,
prolonged qt interval,111975006,LQT,0,4,0,0,118,1391,1513,
qwave abnormal,164917005,QAb,0,1,0,0,548,464,1013,
right axis deviation,47665007,RAD,0,1,0,0,343,83,427,
right bundle branch block,59118001,RBBB,1857,1,2,0,0,542,2402,We score 713427006 and 59118001 as the same diagnosis.
sinus arrhythmia,427393009,SA,0,11,2,0,772,455,1240,
sinus bradycardia,426177001,SB,0,45,0,0,637,1677,2359,
sinus rhythm,426783006,NSR,918,4,0,80,18092,1752,20846,
sinus tachycardia,427084000,STach,0,303,11,1,826,1261,2402,
supraventricular premature beats,63593006,SVPB,0,53,4,0,157,1,215,We score 284470004 and 63593006 as the same diagnosis.
t wave abnormal,164934002,TAb,0,22,0,0,2345,2306,4673,
t wave inversion,59931005,TInv,0,5,1,0,294,812,1112,
ventricular premature beats,17338001,VPB,0,8,0,0,0,357,365,We score 427172004 and 17338001 as the same diagnosis."""))
dx_mapping_scored = dx_mapping_scored.fillna("")
dx_mapping_scored["SNOMED CT Code"] = dx_mapping_scored["SNOMED CT Code"].apply(str)


dx_mapping_unscored = pd.read_csv(StringIO("""Dx,SNOMED CT Code,Abbreviation,CPSC,CPSC-Extra,StPetersburg,PTB,PTB-XL,Georgia,Total
2nd degree av block,195042002,IIAVB,0,21,0,0,14,23,58
abnormal QRS,164951009,abQRS,0,0,0,0,3389,0,3389
accelerated junctional rhythm,426664006,AJR,0,0,0,0,0,19,19
acute myocardial infarction,57054005,AMI,0,0,6,0,0,0,6
acute myocardial ischemia,413444003,AMIs,0,1,0,0,0,1,2
anterior ischemia,426434006,AnMIs,0,0,0,0,44,281,325
anterior myocardial infarction,54329005,AnMI,0,62,0,0,354,0,416
atrial bigeminy,251173003,AB,0,0,3,0,0,0,3
atrial fibrillation and flutter,195080001,AFAFL,0,39,0,0,0,2,41
atrial hypertrophy,195126007,AH,0,2,0,0,0,60,62
atrial pacing pattern,251268003,AP,0,0,0,0,0,52,52
atrial tachycardia,713422000,ATach,0,15,0,0,0,28,43
atrioventricular junctional rhythm,29320008,AVJR,0,6,0,0,0,0,6
av block,233917008,AVB,0,5,0,0,0,74,79
blocked premature atrial contraction,251170000,BPAC,0,2,3,0,0,0,5
brady tachy syndrome,74615001,BTS,0,1,1,0,0,0,2
bundle branch block,6374002,BBB,0,0,1,20,0,116,137
cardiac dysrhythmia,698247007,CD,0,0,0,16,0,0,16
chronic atrial fibrillation,426749004,CAF,0,1,0,0,0,0,1
chronic myocardial ischemia,413844008,CMI,0,161,0,0,0,0,161
complete heart block,27885002,CHB,0,27,0,0,16,8,51
congenital incomplete atrioventricular heart block,204384007,CIAHB,0,0,0,2,0,0,2
coronary heart disease,53741008,CHD,0,0,16,21,0,0,37
decreased qt interval,77867006,SQT,0,1,0,0,0,0,1
diffuse intraventricular block,82226007,DIB,0,1,0,0,0,0,1
early repolarization,428417006,ERe,0,0,0,0,0,140,140
fusion beats,13640000,FB,0,0,7,0,0,0,7
heart failure,84114007,HF,0,0,0,7,0,0,7
heart valve disorder,368009,HVD,0,0,0,6,0,0,6
high t-voltage,251259000,HTV,0,1,0,0,0,0,1
idioventricular rhythm,49260003,IR,0,0,2,0,0,0,2
incomplete left bundle branch block,251120003,ILBBB,0,42,0,0,77,86,205
indeterminate cardiac axis,251200008,ICA,0,0,0,0,156,0,156
inferior ischaemia,425419005,IIs,0,0,0,0,219,451,670
inferior ST segment depression,704997005,ISTD,0,1,0,0,0,0,1
junctional escape,426995002,JE,0,4,0,0,0,5,9
junctional premature complex,251164006,JPC,0,2,0,0,0,0,2
junctional tachycardia,426648003,JTach,0,2,0,0,0,4,6
lateral ischaemia,425623009,LIs,0,0,0,0,142,903,1045
left atrial abnormality,253352002,LAA,0,0,0,0,0,72,72
left atrial enlargement,67741000119109,LAE,0,1,0,0,427,870,1298
left atrial hypertrophy,446813000,LAH,0,40,0,0,0,0,40
left posterior fascicular block,445211001,LPFB,0,0,0,0,177,25,202
left ventricular hypertrophy,164873001,LVH,0,158,10,0,2359,1232,3759
left ventricular strain,370365005,LVS,0,1,0,0,0,0,1
mobitz type i wenckebach atrioventricular block,54016002,MoI,0,0,3,0,0,0,3
myocardial infarction,164865005,MI,0,376,9,368,5261,7,6021
myocardial ischemia,164861001,MIs,0,384,0,0,2175,0,2559
nonspecific st t abnormality,428750005,NSSTTA,0,1290,0,0,381,1883,3554
old myocardial infarction,164867002,OldMI,0,1168,0,0,0,0,1168
paired ventricular premature complexes,251182009,VPVC,0,0,23,0,0,0,23
paroxysmal atrial fibrillation,282825002,PAF,0,0,1,1,0,0,2
paroxysmal supraventricular tachycardia,67198005,PSVT,0,0,3,0,24,0,27
paroxysmal ventricular tachycardia,425856008,PVT,0,0,15,0,0,0,15
r wave abnormal,164921003,RAb,0,1,0,0,0,10,11
rapid atrial fibrillation,314208002,RAF,0,0,0,2,0,0,2
right atrial abnormality,253339007,RAAb,0,0,0,0,0,14,14
right atrial hypertrophy,446358003,RAH,0,18,0,0,99,0,117
right ventricular hypertrophy,89792004,RVH,0,20,0,0,126,86,232
s t changes,55930002,STC,0,1,0,0,770,6,777
shortened pr interval,49578007,SPRI,0,3,0,0,0,2,5
sinoatrial block,65778007,SAB,0,9,0,0,0,0,9
sinus node dysfunction,60423000,SND,0,0,2,0,0,0,2
st depression,429622005,STD,869,57,4,0,1009,38,1977
st elevation,164931005,STE,220,66,4,0,28,134,452
st interval abnormal,164930006,STIAb,0,481,2,0,0,992,1475
supraventricular bigeminy,251168009,SVB,0,0,1,0,0,0,1
supraventricular tachycardia,426761007,SVT,0,3,1,0,27,32,63
suspect arm ecg leads reversed,251139008,ALR,0,0,0,0,0,12,12
transient ischemic attack,266257000,TIA,0,0,7,0,0,0,7
u wave abnormal,164937009,UAb,0,1,0,0,0,0,1
ventricular bigeminy,11157007,VBig,0,5,9,0,82,2,98
ventricular ectopics,164884008,VEB,700,0,49,0,1154,41,1944
ventricular escape beat,75532003,VEsB,0,3,1,0,0,0,4
ventricular escape rhythm,81898007,VEsR,0,1,0,0,0,1,2
ventricular fibrillation,164896001,VF,0,10,0,25,0,3,38
ventricular flutter,111288001,VFL,0,1,0,0,0,0,1
ventricular hypertrophy,266249003,VH,0,5,0,13,30,71,119
ventricular pacing pattern,251266004,VPP,0,0,0,0,0,46,46
ventricular pre excitation,195060002,VPEx,0,6,0,0,0,2,8
ventricular tachycardia,164895002,VTach,0,1,1,10,0,0,12
ventricular trigeminy,251180001,VTrig,0,4,4,0,20,1,29
wandering atrial pacemaker,195101003,WAP,0,0,0,0,0,7,7
wolff parkinson white pattern,74390002,WPW,0,0,4,2,80,2,88"""))
dx_mapping_unscored["SNOMED CT Code"] = dx_mapping_unscored["SNOMED CT Code"].apply(str)


dms = dx_mapping_scored.copy()
dms["scored"] = True
dmn = dx_mapping_unscored.copy()
dmn["Notes"] = ""
dmn["scored"] = False
dx_mapping_all = pd.concat([dms, dmn], ignore_index=True).fillna("")


df_weights_snomed = df_weights_expanded  # alias


snomed_ct_code_to_abbr = \
    ED({row["SNOMED CT Code"]:row["Abbreviation"] for _,row in dx_mapping_all.iterrows()})
abbr_to_snomed_ct_code = ED({v:k for k,v in snomed_ct_code_to_abbr.items()})

df_weights_abbr = df_weights_expanded.copy()

df_weights_abbr.columns = \
    df_weights_abbr.columns.map(lambda i: snomed_ct_code_to_abbr.get(i, i))
    # df_weights_abbr.columns.map(lambda i: snomed_ct_code_to_abbr[i])

df_weights_abbr.index = \
    df_weights_abbr.index.map(lambda i: snomed_ct_code_to_abbr.get(i, i))
    # df_weights_abbr.index.map(lambda i: snomed_ct_code_to_abbr[i])

df_weights_abbreviations = df_weights.copy()  # corresponding to weights_abbreviations.csv
df_weights_abbreviations.columns = \
    df_weights_abbreviations.columns.map(lambda i: "|".join([snomed_ct_code_to_abbr.get(item, item) for item in i.split("|")]))
    # df_weights_abbreviations.columns.map(lambda i: "|".join([snomed_ct_code_to_abbr[item] for item in i.split("|")]))
df_weights_abbreviations.index = \
    df_weights_abbreviations.index.map(lambda i: "|".join([snomed_ct_code_to_abbr.get(item, item) for item in i.split("|")]))
    # df_weights_abbreviations.index.map(lambda i: "|".join([snomed_ct_code_to_abbr[item] for item in i.split("|")]))


snomed_ct_code_to_fullname = \
    ED({row["SNOMED CT Code"]:row["Dx"] for _,row in dx_mapping_all.iterrows()})
fullname_to_snomed_ct_code = ED({v:k for k,v in snomed_ct_code_to_fullname.items()})

df_weights_fullname = df_weights_expanded.copy()

df_weights_fullname.columns = \
    df_weights_fullname.columns.map(lambda i: snomed_ct_code_to_fullname.get(i, i))
    # df_weights_fullname.columns.map(lambda i: snomed_ct_code_to_fullname[i])

df_weights_fullname.index = \
    df_weights_fullname.index.map(lambda i: snomed_ct_code_to_fullname.get(i, i))
    # df_weights_fullname.index.map(lambda i: snomed_ct_code_to_fullname[i])


abbr_to_fullname = \
    ED({row["Abbreviation"]:row["Dx"] for _,row in dx_mapping_all.iterrows()})
fullname_to_abbr = ED({v:k for k,v in abbr_to_fullname.items()})


equiv_class_dict = ED({
    "CRBBB": "RBBB",
    "SVPB": "PAC",
    "VPB": "PVC",
    "713427006": "59118001",
    "63593006": "284470004",
    "17338001": "427172004",
    "complete right bundle branch block": "right bundle branch block",
    "supraventricular premature beats": "premature atrial contraction",
    "ventricular premature beats": "premature ventricular contractions",
})


# functions

def load_weights(classes:Sequence[Union[int,str]]=None,
                 equivalent_classes:Optional[Union[Dict[str,str], List[List[str]]]]=None,
                 return_fmt:str="np") -> Union[np.ndarray, pd.DataFrame]:
    """ NOT finished, NOT checked,

    load the weight matrix of the `classes`

    Parameters:
    -----------
    classes: sequence of str or int, optional,
        the classes (abbr. or SNOMED CT Code) to load their weights,
        if not given, weights of all classes in `dx_mapping_scored` will be loaded
    equivalent_classes: dict or list, optional,
        list or dict of equivalent classes,
        if not specified, defaults to `equiv_class_dict`
    return_fmt: str, default "np",
        "np" or "pd", the values in the form of a 2d array or a DataFrame

    Returns:
    --------
    mat: 2d array or DataFrame,
        the weight matrix of the `classes`
    """
    if classes:
        l_nc = [normalize_class(c, ensure_scored=True) for c in classes]
        assert len(set(l_nc)) == len(classes), "`classes` has duplicates!"
        mat = df_weights_abbr.loc[l_nc,l_nc]
    else:
        mat = df_weights_abbr.copy()
    
    if return_fmt.lower() == "np":
        mat = mat.values
    elif return_fmt.lower() == "pd":
        # columns and indices back to the original input format
        mat.columns = list(map(str, classes))
        mat.index = list(map(str, classes))
    else:
        raise ValueError(f"format of `{return_fmt}` is not supported!")
    
    return mat


def normalize_class(c:Union[str,int], ensure_scored:bool=False) -> str:
    """ finished, checked,

    normalize the class name to its abbr.,
    facilitating the computation of the `load_weights` function

    Parameters:
    -----------
    c: str or int,
        abbr. or SNOMED CT Code of the class
    ensure_scored: bool, default False,
        ensure that the class is a scored class,
        if True, `ValueError` would be raised if `c` is not scored

    Returns:
    --------
    nc: str,
        the abbr. of the class
    """
    nc = snomed_ct_code_to_abbr.get(str(c), str(c))
    if ensure_scored and nc not in df_weights_abbr.columns:
        raise ValueError(f"class `{c}` not among the scored classes")
    return nc


def get_class(snomed_ct_code:Union[str,int]) -> Dict[str,str]:
    """ finished, checked,

    look up the abbreviation and the full name of an ECG arrhythmia,
    given its SNOMED CT Code

    Parameters:
    -----------
    snomed_ct_code: str or int,
        the SNOMED CT Code of the arrhythmia
    
    Returns:
    --------
    arrhythmia_class: dict,
        containing `abbr` the abbreviation and `fullname` the full name of the arrhythmia
    """
    arrhythmia_class = {
        "abbr": snomed_ct_code_to_abbr[str(snomed_ct_code)],
        "fullname": snomed_ct_code_to_fullname[str(snomed_ct_code)],
    }
    return arrhythmia_class


def get_class_count(tranches:Union[str, Sequence[str]],
                    exclude_classes:Optional[Sequence[str]]=None,
                    scored_only:bool=False,
                    normalize:bool=True,
                    threshold:Optional[Real]=0,
                    fmt:str="a") ->Dict[str, int]:
    """ finished, checked,

    Parameters:
    -----------
    tranches: str or sequence of str,
        tranches to count classes, can be combinations of "A", "B", "C", "D", "E", "F"
    exclude_classes: sequence of str, optional,
        abbrevations or SNOMED CT Codes of classes to be excluded from counting
    scored_only: bool, default True,
        if True, only scored classes are counted
    normalize: bool, default True,
        collapse equivalent classes into one,
        used only when `scored_only` = True
    threshold: real number,
        minimum ratio (0-1) or absolute number (>1) of a class to be counted
    fmt: str, default "a",
        the format of the names of the classes in the returned dict,
        can be one of the following (case insensitive):
        - "a", abbreviations
        - "f", full names
        - "s", SNOMED CT Code

    Returns:
    --------
    class_count: dict,
        key: class in the format of `fmt`
        value: count of a class in `tranches`
    """
    assert threshold >= 0
    tranche_names = ED({
        "A": "CPSC",
        "B": "CPSC-Extra",
        "C": "StPetersburg",
        "D": "PTB",
        "E": "PTB-XL",
        "F": "Georgia",
    })
    tranche_names = [tranche_names[t] for t in tranches]
    _exclude_classes = [normalize_class(c) for c in (exclude_classes or [])]
    df = dx_mapping_scored.copy() if scored_only else dx_mapping_all.copy()
    class_count = ED()
    for _, row in df.iterrows():
        key = row["Abbreviation"]
        val = row[tranche_names].values.sum()
        if val == 0:
            continue
        if key in _exclude_classes:
            continue
        if normalize and scored_only:
            key = equiv_class_dict.get(key, key)
        if key in _exclude_classes:
            continue
        if key in class_count.keys():
            class_count[key] += val
        else:
            class_count[key] = val
    tmp = ED()
    tot_count = sum(class_count.values())
    _threshold = threshold if threshold >= 1 else threshold * tot_count
    if fmt.lower() == "s":
        for key, val in class_count.items():
            if val < _threshold:
                continue
            tmp[abbr_to_snomed_ct_code[key]] = val
        class_count = tmp.copy()
    elif fmt.lower() == "f":
        for key, val in class_count.items():
            if val < _threshold:
                continue
            tmp[abbr_to_fullname[key]] = val
        class_count = tmp.copy()
    else:
        class_count = {key: val for key, val in class_count.items() if val >= _threshold}
    del tmp
    return class_count


def get_class_weight(tranches:Union[str, Sequence[str]],
                     exclude_classes:Optional[Sequence[str]]=None,
                     scored_only:bool=False,
                     normalize:bool=True,
                     threshold:Optional[Real]=0,
                     fmt:str="a",
                     min_weight:Real=0.5) ->Dict[str, int]:
    """ finished, checked,

    Parameters:
    -----------
    tranches: str or sequence of str,
        tranches to count classes, can be combinations of "A", "B", "C", "D", "E", "F"
    exclude_classes: sequence of str, optional,
        abbrevations or SNOMED CT Codes of classes to be excluded from counting
    scored_only: bool, default True,
        if True, only scored classes are counted
    normalize: bool, default True,
        collapse equivalent classes into one,
        used only when `scored_only` = True
    threshold: real number,
        minimum ratio (0-1) or absolute number (>1) of a class to be counted
    fmt: str, default "a",
        the format of the names of the classes in the returned dict,
        can be one of the following (case insensitive):
        - "a", abbreviations
        - "f", full names
        - "s", SNOMED CT Code
    min_weight: real number, default 0.5,
        minimum value of the weight of all classes,
        or equivalently the weight of the largest class

    Returns:
    --------
    class_weight: dict,
        key: class in the format of `fmt`
        value: weight of a class in `tranches`
    """
    class_count = get_class_count(
        tranches=tranches,
        exclude_classes=exclude_classes,
        scored_only=scored_only,
        normalize=normalize,
        threshold=threshold,
        fmt=fmt,
    )
    class_weight = ED({
        key: sum(class_count.values()) / val for key, val in class_count.items()
    })
    class_weight = ED({
        key: min_weight * val / min(class_weight.values()) for key, val in class_weight.items()
    })
    return class_weight


# extra statistics

dx_cooccurrence_all = pd.read_csv(StringIO(""",IAVB,AF,AFL,Brady,CRBBB,IRBBB,LAnFB,LAD,LBBB,LQRSV,NSIVCB,PR,PAC,PVC,LPR,LQT,QAb,RAD,RBBB,SA,SB,NSR,STach,SVPB,TAb,TInv,VPB,IIAVB,abQRS,AJR,AMI,AMIs,AnMIs,AnMI,AB,AFAFL,AH,AP,ATach,AVJR,AVB,BPAC,BTS,BBB,CD,CAF,CMI,CHB,CIAHB,CHD,SQT,DIB,ERe,FB,HF,HVD,HTV,IR,ILBBB,ICA,IIs,ISTD,JE,JPC,JTach,LIs,LAA,LAE,LAH,LPFB,LVH,LVS,MoI,MI,MIs,NSSTTA,OldMI,VPVC,PAF,PSVT,PVT,RAb,RAF,RAAb,RAH,RVH,STC,SPRI,SAB,SND,STD,STE,STIAb,SVB,SVT,ALR,TIA,UAb,VBig,VEB,VEsB,VEsR,VF,VFL,VH,VPP,VPEx,VTach,VTrig,WAP,WPW
IAVB,2394,24,7,16,85,77,148,469,158,15,92,0,77,8,125,119,61,32,84,58,251,614,89,17,223,67,43,3,177,0,0,0,22,23,0,0,3,0,4,0,0,2,1,19,0,0,6,0,0,0,0,0,7,0,0,0,0,0,30,5,29,0,0,0,0,87,6,179,2,24,202,0,0,391,121,166,25,0,0,0,0,0,0,0,7,15,18,0,0,0,38,17,88,0,2,3,0,0,4,93,0,1,0,0,13,2,0,0,1,1,0
AF,24,3475,32,4,104,139,148,528,124,36,102,4,20,19,0,102,68,70,244,2,17,37,13,4,455,110,20,6,313,2,0,0,17,32,0,16,0,3,10,4,9,0,0,11,8,1,10,4,0,3,0,0,3,0,1,0,0,0,29,12,92,0,1,0,1,103,1,7,0,40,355,0,0,606,452,330,79,0,0,1,0,0,0,0,0,10,217,0,0,0,252,10,79,0,1,2,0,0,9,215,0,0,4,0,8,9,0,1,0,1,2
AFL,7,32,314,0,6,11,5,40,3,13,13,0,7,2,1,26,13,6,12,1,4,5,11,4,69,21,5,2,12,0,0,0,11,0,0,0,0,0,0,1,65,0,0,1,1,0,2,0,0,0,0,0,4,0,0,0,0,0,3,0,23,0,0,0,0,16,0,4,0,1,28,0,0,10,7,64,4,0,0,0,0,0,0,0,0,1,11,0,0,0,4,5,14,0,7,1,0,0,0,1,0,0,7,0,0,1,0,0,0,0,0
Brady,16,4,0,288,10,15,0,0,2,1,1,0,5,14,0,1,0,0,1,0,1,0,0,5,4,2,1,2,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,12,9,0,4,0,0,0,0,0,0,0,0,3,0,1,0,1,0,0,0,0,0,1,0,23,0,0,16,8,111,107,3,0,0,1,0,0,0,0,0,0,0,2,0,2,9,49,0,0,0,1,0,1,10,1,0,0,0,0,0,0,0,0,0,1
CRBBB,85,104,6,10,683,54,162,219,3,0,1,0,30,8,28,1,43,74,14,13,14,364,33,14,9,4,4,2,149,1,0,0,0,6,0,1,0,0,1,1,2,0,0,0,0,0,4,5,0,0,0,0,0,0,0,0,0,0,0,8,11,0,0,0,0,5,0,30,1,86,12,0,0,291,19,43,63,0,0,0,0,0,0,0,3,56,14,0,0,0,14,3,14,0,2,0,0,0,5,40,0,0,0,0,0,0,0,0,0,0,0
IRBBB,77,139,11,15,54,1611,136,340,2,4,14,2,70,6,24,76,45,74,62,69,106,873,109,10,176,39,15,8,160,1,0,0,32,29,0,1,1,9,2,2,1,0,0,0,0,0,1,5,0,0,0,0,0,0,0,0,0,0,1,11,32,0,1,0,0,24,0,75,1,36,92,0,0,258,79,135,41,0,0,1,0,0,0,0,16,81,25,0,0,0,36,5,83,0,4,0,0,0,4,71,0,0,1,0,19,0,0,0,1,0,3
LAnFB,148,148,5,0,162,136,1806,1386,1,9,30,1,78,0,62,39,69,16,9,48,53,1241,107,21,169,42,13,1,353,0,0,1,12,44,0,0,1,1,1,0,0,0,0,3,0,0,0,1,0,0,0,0,1,0,0,0,0,0,5,14,20,0,0,0,0,37,2,73,0,3,184,0,0,718,193,66,0,0,0,0,0,0,0,0,9,25,50,0,0,0,88,5,17,0,1,0,0,0,8,139,0,0,0,0,7,2,0,0,2,0,0
LAD,469,528,40,0,219,340,1386,6086,405,52,246,7,138,0,81,134,148,0,100,208,320,4052,354,76,574,110,54,12,1630,3,0,0,37,125,0,0,0,11,6,0,1,0,0,66,0,0,0,5,0,0,0,0,4,0,0,0,0,0,50,0,50,0,0,0,1,201,6,221,0,8,683,0,0,2227,580,191,0,0,0,2,0,0,0,1,16,41,146,0,0,0,132,21,72,0,8,0,0,0,32,385,0,0,0,0,23,4,0,0,4,1,41
LBBB,158,124,3,2,3,2,1,405,1041,1,4,1,30,4,16,6,5,1,0,17,45,371,55,10,10,23,7,1,12,1,0,0,1,2,0,1,2,1,2,0,0,0,0,91,0,0,1,4,0,0,0,0,0,0,0,0,0,0,15,11,3,0,0,0,0,12,2,93,2,1,31,0,0,41,12,32,20,0,0,0,0,0,0,0,2,2,4,0,0,0,4,4,12,0,2,1,0,0,2,62,0,0,0,0,2,0,0,0,0,0,0
LQRSV,15,36,13,1,0,4,9,52,1,556,18,1,40,0,5,36,4,2,1,16,59,154,87,0,128,24,9,0,12,1,0,0,15,0,0,0,0,0,2,0,6,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,6,18,0,0,0,1,34,0,8,0,0,2,0,0,47,9,83,0,0,0,0,0,0,0,0,2,1,13,0,0,0,9,5,25,0,5,1,0,0,0,10,0,0,0,0,0,1,0,0,0,0,0
NSIVCB,92,102,13,1,1,14,30,246,4,18,997,1,39,1,23,35,83,31,2,24,70,589,56,4,160,46,8,2,186,0,0,0,8,32,0,0,0,3,2,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,2,7,23,0,0,0,0,49,3,62,1,11,154,0,0,353,180,53,0,0,0,0,0,0,0,0,7,1,35,1,0,0,86,2,26,0,3,0,0,0,4,81,0,1,0,0,1,0,0,0,0,0,0
PR,0,4,0,0,0,2,1,7,1,1,1,299,1,0,0,0,0,3,0,0,2,5,0,0,0,1,0,0,2,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,0,0,5,2,4,1,0,0,0,0,0,0,0,0,1,1,0,0,0,2,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0
PAC,77,20,7,5,30,70,78,138,30,40,39,1,1729,14,24,96,46,4,89,16,100,351,140,2,229,77,27,2,13,0,0,0,16,9,3,0,6,1,2,0,1,3,0,8,0,0,1,2,0,3,0,0,1,1,0,0,0,0,12,8,31,0,1,0,0,70,3,63,0,4,153,0,0,120,63,188,17,1,0,1,1,0,0,1,4,6,26,0,0,0,76,11,69,0,1,0,0,0,22,61,0,0,0,0,8,1,0,0,5,1,1
PVC,8,19,2,14,8,6,0,0,4,0,1,0,14,188,0,0,0,1,0,1,4,0,15,1,0,0,2,0,0,0,0,0,0,6,0,1,0,0,2,1,2,0,0,0,0,0,5,4,0,0,0,1,0,0,0,0,0,0,3,0,0,0,2,0,1,0,0,0,2,0,10,0,0,39,14,78,49,0,0,0,0,0,0,0,2,2,0,0,3,0,3,1,12,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0
LPR,125,0,1,0,28,24,62,81,16,5,23,0,24,0,340,0,37,1,0,5,8,319,13,0,75,17,0,0,3,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,1,0,0,0,0,2,0,35,0,5,66,0,0,131,60,15,0,0,0,0,0,0,0,0,1,1,12,0,0,0,59,0,0,0,0,0,0,0,1,36,0,0,0,0,0,0,0,0,2,0,0
LQT,119,102,26,1,1,76,39,134,6,36,35,0,96,0,0,1513,112,18,13,37,104,96,75,2,521,140,68,5,13,7,0,1,61,2,0,0,13,12,3,0,9,0,0,1,0,0,0,1,0,0,0,0,4,0,0,0,0,0,22,1,101,0,1,0,1,147,11,187,0,5,231,0,0,23,11,403,0,0,0,0,0,2,0,3,3,25,9,0,0,0,15,23,936,0,5,2,0,0,0,15,0,0,2,0,18,2,0,0,1,0,0
QAb,61,68,13,0,43,45,69,148,5,4,83,0,46,0,37,112,1013,16,18,29,88,452,73,0,458,138,16,1,2,1,0,0,86,23,0,0,0,1,1,0,2,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,5,10,124,0,0,0,0,170,1,40,0,8,127,0,0,421,84,65,1,0,0,0,0,0,0,0,1,3,16,0,0,0,102,0,84,0,2,1,0,0,1,38,0,0,0,0,12,1,1,0,1,0,0
RAD,32,70,6,0,74,74,16,0,1,2,31,3,4,1,1,18,16,427,1,24,10,226,46,5,34,9,6,1,145,0,0,0,7,8,0,0,1,2,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,19,0,0,0,0,3,1,22,0,102,12,0,0,156,26,33,0,0,0,1,0,0,0,2,18,47,4,1,0,0,6,2,11,0,0,3,0,0,3,30,0,0,0,0,13,0,0,0,0,0,3
RBBB,84,244,12,1,14,62,9,100,0,1,2,0,89,0,0,13,18,1,2402,16,88,0,61,0,72,47,18,2,0,1,0,0,13,0,0,0,4,6,5,0,1,0,0,25,0,0,0,3,0,0,0,0,2,0,0,0,0,0,2,0,42,0,0,0,1,70,1,53,0,4,36,0,0,1,0,45,1,0,0,0,0,0,0,1,0,10,0,0,0,0,22,22,17,0,0,1,2,0,1,54,0,0,2,0,4,1,0,0,0,2,0
SA,58,2,1,0,13,69,48,208,17,16,24,0,16,1,5,37,29,24,16,1240,123,428,0,1,132,27,6,1,128,0,0,0,10,9,0,0,2,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,11,0,0,0,1,0,3,1,19,0,0,0,0,34,0,40,0,6,114,0,0,158,53,82,4,0,0,0,0,0,0,0,2,7,15,0,0,0,16,7,27,0,1,2,1,0,3,123,0,0,0,0,4,1,0,0,0,0,3
SB,251,17,4,1,14,106,53,320,45,59,70,2,100,4,8,104,88,10,88,123,2359,353,1,1,318,114,25,6,85,0,0,0,47,9,0,0,0,1,0,0,2,0,0,9,0,0,3,5,0,0,0,0,43,0,0,0,0,0,12,9,45,0,2,0,0,110,12,106,0,5,265,0,0,109,56,257,12,0,0,0,0,0,0,0,1,10,20,1,0,0,30,21,70,0,0,0,0,0,3,6,1,0,0,0,7,2,1,0,0,1,3
NSR,614,37,5,0,364,873,1241,4052,371,154,589,5,351,0,319,96,452,226,0,428,353,20846,236,36,1784,223,0,1,2629,0,0,0,22,265,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,47,134,142,0,0,0,0,87,0,343,0,107,1721,0,0,3789,1385,220,2,0,0,2,0,0,0,0,62,103,410,1,0,0,688,23,0,0,3,0,0,0,38,662,0,0,0,0,23,0,0,0,11,0,66
STach,89,13,11,0,33,109,107,354,55,87,56,0,140,15,13,75,73,46,61,0,1,236,2402,8,409,130,67,9,151,0,0,0,44,28,0,2,26,3,5,0,1,0,1,13,0,0,13,1,0,5,0,0,8,5,0,1,0,0,24,8,76,0,2,0,2,111,24,195,9,17,241,0,1,302,120,424,165,4,0,1,2,1,0,7,34,24,60,1,0,0,90,36,84,0,3,2,0,0,13,108,0,0,0,0,15,4,0,0,6,0,0
SVPB,17,4,4,5,14,10,21,76,10,0,4,0,2,1,0,2,0,5,0,1,1,36,8,215,13,0,0,0,50,0,0,0,1,5,0,2,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,3,0,2,0,1,0,0,3,0,4,3,3,23,0,0,65,24,25,8,2,0,3,0,0,0,0,2,3,3,0,0,0,1,0,5,1,0,0,0,0,3,11,1,0,0,0,0,0,0,1,2,0,6
TAb,223,455,69,4,9,176,169,574,10,128,160,0,229,0,75,521,458,34,72,132,318,1784,409,13,4673,413,88,11,85,10,0,0,255,37,0,2,18,14,14,0,27,0,0,2,0,0,0,6,0,0,0,0,16,0,0,0,0,0,34,21,412,0,0,0,1,692,5,226,0,12,536,0,0,348,163,418,12,0,0,1,0,2,0,2,5,22,63,0,0,0,370,30,364,0,16,3,0,0,7,184,0,0,0,0,23,5,0,0,4,3,4
TInv,67,110,21,2,4,39,42,110,23,24,46,1,77,0,17,140,138,9,47,27,114,223,130,0,413,1112,34,1,0,4,1,0,45,12,0,1,7,3,3,0,8,0,0,11,0,0,0,1,0,0,0,0,9,0,0,0,0,0,11,4,70,0,0,0,0,129,2,95,0,6,226,0,0,179,151,391,2,0,0,0,0,0,0,0,2,9,1,0,0,0,157,22,78,0,6,2,0,0,3,34,0,0,0,0,13,0,0,0,0,0,0
VPB,43,20,5,1,4,15,13,54,7,9,8,0,27,2,0,68,16,6,18,6,25,0,67,0,88,34,365,1,0,3,0,0,5,0,0,0,2,1,2,0,2,0,0,5,0,0,0,0,0,0,0,0,3,0,0,0,0,0,11,0,15,0,1,0,0,40,2,42,0,1,43,0,0,4,0,93,1,0,0,0,0,0,0,1,0,5,0,0,0,0,1,7,48,0,1,1,0,0,1,3,0,0,0,0,6,7,0,0,0,0,0
IIAVB,3,6,2,2,2,8,1,12,1,0,2,0,2,0,0,5,1,1,2,1,6,1,9,0,11,1,1,58,2,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,4,0,0,1,1,9,0,0,1,6,7,1,0,0,0,0,0,0,0,0,1,2,0,0,0,0,0,4,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0
abQRS,177,313,12,0,149,160,353,1630,12,12,186,2,13,0,3,13,2,145,0,128,85,2629,151,50,85,0,0,2,3389,0,0,0,12,158,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,19,1,26,0,0,0,0,55,0,86,0,44,383,0,0,2644,464,45,0,0,0,2,0,0,0,0,19,26,37,0,0,0,6,0,0,0,0,0,0,0,6,251,0,0,0,0,6,0,0,0,3,0,41
AJR,0,2,0,0,1,1,0,3,1,1,0,0,0,0,0,7,1,0,1,0,0,0,0,0,10,4,3,0,0,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,4,0,0,0,0,2,0,0,0,2,2,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0
AMI,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0
AMIs,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
AnMIs,22,17,11,1,0,32,12,37,1,15,8,0,16,0,0,61,86,7,13,10,47,22,44,1,255,45,5,0,12,0,0,0,325,2,0,0,1,4,2,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,27,0,0,0,1,17,2,24,0,0,32,0,0,27,6,32,0,0,0,0,0,0,0,1,0,1,0,0,0,0,3,2,40,0,0,0,0,0,0,2,0,0,0,0,5,1,0,0,0,0,0
AnMI,23,32,0,1,6,29,44,125,2,0,32,1,9,6,4,2,23,8,0,9,9,265,28,5,37,12,0,0,158,0,0,0,2,416,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,3,4,4,0,0,0,0,6,0,8,1,8,45,0,0,159,55,26,2,0,0,0,0,0,0,0,5,4,7,0,1,0,31,14,13,0,0,0,0,0,5,24,0,0,0,0,0,0,0,0,0,0,0
AB,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
AFAFL,0,16,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,2,2,2,1,0,0,0,0,0,0,0,0,0,41,0,0,2,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,5,0,0,0,0,13,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0
AH,3,0,0,0,0,1,1,0,2,0,0,0,6,0,0,13,0,1,4,2,0,0,26,1,18,7,2,0,0,0,0,0,1,0,0,0,62,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,4,0,0,0,1,7,0,0,0,1,17,0,0,0,0,16,1,0,0,0,0,0,0,0,0,5,0,0,0,0,2,1,8,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0
AP,0,3,0,0,0,9,1,11,1,0,3,0,1,0,0,12,1,2,6,1,1,0,3,0,14,3,1,0,0,0,0,0,4,0,0,0,0,52,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,0,2,4,0,0,0,0,8,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,10,0,0,0,0,0,0,0,0,0,0,0,2,2,0,0,0,1,0
ATach,4,10,0,0,1,2,1,6,2,2,2,0,2,2,0,3,1,0,5,0,0,0,5,0,14,3,2,0,0,0,0,0,2,1,0,2,0,0,43,0,1,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,5,0,0,0,0,1,1,1,0,0,3,0,0,0,0,14,5,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,4,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
AVJR,0,4,1,0,1,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
AVB,0,9,65,1,2,1,0,1,0,6,0,0,1,2,0,9,2,1,1,0,2,0,1,0,27,8,2,1,0,0,0,0,1,0,0,0,0,0,1,0,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,3,0,0,0,0,12,0,0,0,0,27,2,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
BPAC,2,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,5,0,0,0,0,0,0,0,3,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0
BTS,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
BBB,19,11,1,0,0,0,3,66,91,0,0,0,8,0,0,1,0,1,25,1,9,0,13,0,2,11,5,0,0,0,0,0,0,0,0,0,2,2,2,0,0,0,0,137,0,0,0,0,0,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,2,43,0,0,0,0,0,2,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1
CD,0,8,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,0,0,0,2,3,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
CAF,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
CMI,6,10,2,12,4,1,0,0,1,0,0,0,1,5,0,0,0,0,0,1,3,0,13,1,0,0,0,3,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,161,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,65,2,22,82,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0
CHB,0,4,0,9,5,5,1,5,4,2,0,0,2,4,0,1,3,1,3,0,5,3,1,0,6,1,0,0,4,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,0,0,0,0,0,0,0,0,0,0,3,0,0,0,2,0,0,2,0,1,0,1,3,0,0,10,5,8,2,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0
CIAHB,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
CHD,0,3,0,4,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,5,1,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,3,0,2,3,0,0,0,0,37,0,0,0,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,9,0,0,0,8,0,1,2,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,0,1,0,2,0,0,2,1,0,0
SQT,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
DIB,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
ERe,7,3,4,0,0,0,1,4,0,0,1,0,1,0,0,4,0,1,2,11,43,0,8,0,16,9,3,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,140,0,0,0,0,0,0,0,2,0,0,0,0,2,0,8,0,1,35,0,0,0,0,12,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,36,3,0,0,1,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0
FB,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,4,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,1,0,0,0,4,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0
HF,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0
HVD,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
HTV,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
IR,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0
ILBBB,30,29,3,3,0,1,5,50,15,0,2,0,12,3,3,22,5,0,2,3,12,47,24,3,34,11,11,3,19,1,0,0,1,3,0,0,0,0,2,0,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,205,0,4,0,1,0,0,28,0,16,0,0,33,0,0,41,31,43,16,0,0,0,0,0,0,0,0,0,2,1,0,0,3,2,21,0,0,0,0,0,0,13,0,0,0,0,3,0,0,0,0,0,0
ICA,5,12,0,0,8,11,14,0,11,6,7,1,8,0,3,1,10,0,0,1,9,134,8,0,21,4,0,0,1,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,156,0,0,0,0,0,0,0,7,0,0,21,0,0,25,12,8,0,0,0,0,0,0,0,0,3,2,6,0,0,0,11,3,0,0,0,0,0,0,3,7,0,0,0,0,1,0,0,0,0,0,0
IIs,29,92,23,1,11,32,20,50,3,18,23,0,31,0,1,101,124,19,42,19,45,142,76,2,412,70,15,0,26,4,0,0,27,4,0,0,4,1,5,0,11,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,4,0,670,0,1,0,0,22,0,39,0,12,73,0,0,39,73,44,0,0,0,0,0,0,0,1,3,14,3,0,0,0,23,3,74,0,2,3,0,0,1,16,0,0,0,0,6,3,0,0,0,1,1
ISTD,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
JE,0,1,0,1,0,1,0,0,0,0,0,0,1,2,0,1,0,0,0,0,2,0,2,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,1,0,1,0,9,0,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
JPC,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
JTach,0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,0,0,2,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,1,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
LIs,87,103,16,0,5,24,37,201,12,34,49,0,70,0,2,147,170,3,70,34,110,87,111,3,692,129,40,4,55,2,0,0,17,6,0,1,7,4,1,0,3,0,0,2,0,0,0,2,0,0,0,0,2,0,0,0,0,0,28,0,22,0,0,0,0,1045,6,97,0,1,132,0,0,89,12,73,0,0,0,0,0,0,0,1,1,8,1,0,0,0,12,9,104,0,3,2,0,0,1,22,0,0,0,0,5,3,0,0,0,1,0
LAA,6,1,0,0,0,0,2,6,2,0,3,0,3,0,0,11,1,1,1,0,12,0,24,0,5,2,2,0,0,0,0,0,2,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,72,0,0,1,13,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,5,3,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0
LAE,179,7,4,0,30,75,73,221,93,8,62,0,63,0,35,187,40,22,53,40,106,343,195,4,226,95,42,0,86,0,0,1,24,8,0,0,0,0,1,0,0,0,0,43,0,0,0,1,0,0,0,0,8,0,0,0,0,0,16,7,39,0,0,0,0,97,0,1298,0,12,189,1,0,162,97,200,0,0,0,0,0,1,0,0,16,9,19,0,0,0,52,7,156,0,1,0,0,0,1,56,0,0,0,0,34,0,0,0,0,0,0
LAH,2,0,0,1,1,1,0,0,2,0,1,0,0,2,0,0,0,0,0,0,0,0,9,3,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,40,0,5,0,0,9,3,17,21,0,0,0,0,0,0,0,3,4,0,0,0,0,1,1,3,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0
LPFB,24,40,1,0,86,36,3,8,1,0,11,0,4,0,5,5,8,102,4,6,5,107,17,3,12,6,1,1,44,2,0,0,0,8,0,0,1,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,12,0,0,0,0,1,1,12,0,202,5,0,0,79,9,10,0,0,0,0,0,0,0,0,12,29,6,0,0,0,7,0,3,0,0,1,0,0,0,11,0,0,0,0,0,1,0,0,0,0,0
LVH,202,355,28,23,12,92,184,683,31,2,154,2,153,10,66,231,127,12,36,114,265,1721,241,23,536,226,43,9,383,2,0,0,32,45,3,5,17,4,3,2,12,4,0,0,0,1,0,3,0,10,0,0,35,3,0,0,0,0,33,21,73,1,0,0,0,132,13,189,5,5,3759,1,0,691,1225,400,23,4,0,1,2,1,0,1,6,18,100,1,2,0,404,39,178,0,6,0,0,0,11,219,0,0,0,0,60,2,0,0,3,0,1
LVS,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
MoI,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0
MI,391,606,10,16,291,258,718,2227,41,47,353,5,120,39,131,23,421,156,1,158,109,3789,302,65,348,179,4,1,2644,0,0,0,27,159,0,0,0,0,0,0,0,0,0,2,0,0,65,10,0,9,0,0,0,1,0,0,0,0,41,25,39,0,0,0,0,89,0,162,9,79,691,0,0,6021,805,195,60,2,1,0,3,1,2,0,30,22,105,0,3,0,370,20,21,0,5,0,0,0,24,481,0,0,25,0,6,0,0,8,6,0,7
MIs,121,452,7,8,19,79,193,580,12,9,180,2,63,14,60,11,84,26,0,53,56,1385,120,24,163,151,0,6,464,0,0,0,6,55,0,0,0,0,0,0,0,0,0,0,0,0,2,5,0,0,0,0,0,0,0,0,0,0,31,12,73,0,0,0,1,12,0,97,3,9,1225,0,0,805,2559,183,69,0,0,0,0,0,0,0,16,6,45,0,0,0,462,2,0,0,6,0,0,0,12,212,0,0,1,0,9,0,0,0,3,0,0
NSSTTA,166,330,64,111,43,135,66,191,32,83,53,4,188,78,15,403,65,33,45,82,257,220,424,25,418,391,93,7,45,8,0,1,32,26,0,13,16,8,14,3,27,0,0,1,0,1,22,8,0,0,0,0,12,0,0,0,0,0,43,8,44,0,5,1,2,73,9,200,17,10,400,1,0,195,183,3554,212,0,0,1,0,2,0,1,11,26,32,3,2,0,150,33,277,0,12,1,0,1,3,34,1,0,1,0,17,2,4,1,2,1,0
OldMI,25,79,4,107,63,41,0,0,20,0,0,1,17,49,0,0,1,0,1,4,12,2,165,8,12,2,1,1,0,0,0,0,0,2,0,12,1,0,5,0,2,0,0,0,0,0,82,2,0,0,1,0,0,0,0,0,0,0,16,0,0,1,0,0,2,0,0,0,21,0,23,0,0,60,69,212,1168,0,0,0,0,0,0,0,11,12,0,0,0,0,20,7,23,0,1,0,0,0,1,0,0,0,1,0,2,0,1,0,2,0,0
VPVC,0,0,0,3,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,4,2,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,8,0,0,0,4,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,4,0,1,2,0,0,0,23,0,3,6,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,3,13,1,0,0,0,0,0,0,0,1,0,1
PAF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
PSVT,0,1,0,0,0,1,0,2,0,0,0,0,1,0,0,0,0,1,0,0,0,2,1,3,1,0,0,0,2,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,3,0,27,0,0,0,0,0,0,6,0,0,1,1,0,0,0,9,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0
PVT,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,2,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3,0,0,0,6,0,0,15,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,2,8,0,0,0,0,0,0,0,0,0,0,0
RAb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,2,0,0,0,0,0,11,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
RAF,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
RAAb,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,3,0,2,1,0,0,0,7,0,2,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,14,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0
RAH,7,0,0,0,3,16,9,16,2,2,7,0,4,2,1,3,1,18,0,2,1,62,34,2,5,2,0,0,19,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,0,1,0,16,3,12,6,0,0,30,16,11,11,0,0,0,0,0,0,0,117,21,4,0,0,0,10,0,0,0,0,0,0,0,1,5,0,0,0,0,0,0,0,0,0,0,0
RVH,15,10,1,0,56,81,25,41,2,1,1,1,6,2,1,25,3,47,10,7,10,103,24,3,22,9,5,1,26,0,0,0,1,4,0,0,5,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,14,0,1,0,0,8,0,9,4,29,18,0,0,22,6,26,12,0,0,0,0,0,0,0,21,232,0,0,0,0,1,1,20,0,0,1,0,0,0,4,0,0,0,0,17,0,0,0,0,0,0
STC,18,217,11,0,14,25,50,146,4,13,35,1,26,0,12,9,16,4,0,15,20,410,60,3,63,1,0,2,37,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,2,6,3,0,0,0,0,1,0,19,0,6,100,0,0,105,45,32,0,0,0,6,0,0,0,0,4,0,777,0,0,0,78,3,4,0,15,0,0,0,4,57,0,0,0,0,1,0,0,0,2,0,0
SPRI,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,5,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
SAB,0,0,0,2,0,0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,3,0,2,0,0,0,0,0,0,0,0,0,0,0,0,9,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
SND,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
STD,38,252,4,2,14,36,88,132,4,9,86,2,76,3,59,15,102,6,22,16,30,688,90,1,370,157,1,0,6,0,0,0,3,31,0,0,2,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,3,11,23,1,0,0,2,12,1,52,1,7,404,0,0,370,462,150,20,1,0,1,0,0,0,0,10,1,78,0,0,0,1977,15,6,0,4,0,1,0,3,112,0,0,0,0,2,0,0,0,1,0,2
STE,17,10,5,9,3,5,5,21,4,5,2,1,11,1,0,23,0,2,22,7,21,23,36,0,30,22,7,0,0,0,2,0,2,14,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,36,0,0,0,0,1,2,3,3,0,1,0,0,9,5,7,1,0,39,0,0,20,2,33,7,1,0,0,0,1,0,0,0,1,3,0,1,0,15,452,18,0,0,0,0,0,0,4,0,0,0,0,4,1,0,0,0,0,0
STIAb,88,79,14,49,14,83,17,72,12,25,26,0,69,12,0,936,84,11,17,27,70,0,84,5,364,78,48,4,0,6,0,1,40,13,0,1,8,10,4,0,5,1,1,0,0,0,0,1,0,0,0,1,3,0,0,0,1,0,21,0,74,0,0,0,1,104,3,156,3,3,178,0,0,21,0,277,23,0,0,0,0,0,0,3,0,20,4,1,0,0,6,18,1475,0,0,2,2,0,1,7,0,0,0,0,7,2,0,0,1,0,0
SVB,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0
SVT,2,1,7,0,2,4,1,8,2,5,3,0,1,0,0,5,2,0,0,1,0,3,3,0,16,6,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,3,0,1,0,0,6,0,0,5,6,12,1,0,0,9,0,0,0,0,0,0,15,0,0,0,4,0,0,0,63,0,1,0,0,7,0,0,0,0,0,0,0,0,0,0,0
ALR,3,2,1,0,0,0,0,0,1,1,0,0,0,0,0,2,1,3,1,2,0,0,2,0,3,2,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,3,0,0,0,0,2,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,0,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
TIA,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,2,0,1,0,7,0,1,7,0,0,0,0,0,0,0,0,0,0,0
UAb,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
VBig,4,9,0,1,5,4,8,32,2,0,4,0,22,0,1,0,1,3,1,3,3,38,13,3,7,3,1,0,6,0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,3,1,0,0,0,0,1,0,1,1,0,11,0,0,24,12,3,1,3,0,0,2,0,0,0,1,0,4,0,0,0,3,0,1,0,0,0,1,0,98,48,0,0,0,0,0,0,0,0,2,0,0
VEB,93,215,1,10,40,71,139,385,62,10,81,2,61,0,36,15,38,30,54,123,6,662,108,11,184,34,3,2,251,0,4,0,2,24,1,0,0,0,1,0,0,1,1,0,0,0,0,0,0,10,0,0,0,4,0,0,0,1,13,7,16,0,0,0,0,22,0,56,0,11,219,0,1,481,212,34,0,13,0,2,8,0,0,0,5,4,57,0,0,0,112,4,7,1,7,0,7,0,48,1944,0,0,0,0,0,0,0,0,10,0,2
VEsB,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0
VEsR,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0
VF,0,4,7,0,0,1,0,0,0,0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,25,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,38,0,1,0,0,1,0,0,0
VFL,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0
VH,13,8,0,0,0,19,7,23,2,0,1,0,8,0,0,18,12,13,4,4,7,23,15,0,23,13,6,0,6,0,0,0,5,0,0,2,2,2,0,0,0,0,0,0,0,0,1,0,0,2,0,0,2,0,2,0,0,0,3,1,6,0,0,0,0,5,1,34,1,0,60,0,0,6,9,17,2,0,0,0,0,0,0,0,0,17,1,0,0,0,2,4,7,0,0,0,0,0,0,0,0,0,1,0,119,0,0,0,0,0,0
VPP,2,9,1,0,0,0,2,4,0,1,0,0,1,0,0,2,1,0,1,1,2,0,4,0,5,0,7,0,0,2,0,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,0,3,0,0,0,1,2,0,0,0,0,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,46,0,0,0,0,0
VPEx,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,0,0,0
VTach,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,12,0,0,0
VTrig,1,0,0,0,0,1,2,4,0,0,0,0,5,0,2,1,1,0,0,0,0,11,6,2,4,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,6,3,2,2,1,0,0,0,0,0,0,0,0,2,0,0,0,1,0,1,0,0,0,0,0,2,10,0,0,0,0,0,0,0,0,29,0,0
WAP,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,2,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0
WPW,0,2,0,1,0,3,0,41,0,0,0,0,1,0,0,0,0,3,0,3,3,66,0,6,4,0,0,0,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,7,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,88
"""), index_col=0)
dx_cooccurrence_scored = dx_cooccurrence_all.loc[dx_mapping_scored.Abbreviation, dx_mapping_scored.Abbreviation]


def get_cooccurrence(c1:Union[str,int], c2:Union[str,int], ensure_scored:bool=False) -> int:
    """ finished, checked,

    Parameters:
    -----------
    c1, c2: str or int,
        the 2 classes
    ensure_scored: bool, default False,
        ensure that the class is a scored class,
        if True, `ValueError` would be raised if `c` is not scored

    Returns:
    --------
    cooccurrence: int,
        cooccurrence of class `c1` and `c2`, if they are not the same class;
        otherwise the occurrence of the class `c1` (also `c2`)
    """
    _c1 = normalize_class(c1, ensure_scored=ensure_scored)
    _c2 = normalize_class(c2, ensure_scored=ensure_scored)
    cooccurrence = dx_cooccurrence_all.loc[_c1, _c2]
    return cooccurrence


"""
dx_cooccurrence_all is obtained via the following code

>>> db_dir = "/media/cfs/wenhao71/data/cinc2021_data/"
>>> working_dir = "./working_dir"
>>> dr = CINC2021Reader(db_dir=db_dir,working_dir=working_dir)
>>> dx_cooccurrence_all = pd.DataFrame(np.zeros((len(dx_mapping_all.Abbreviation), len(dx_mapping_all.Abbreviation)),dtype=int), columns=dx_mapping_all.Abbreviation.values)
>>> dx_cooccurrence_all.index = dx_mapping_all.Abbreviation.values
>>> for tranche, l_rec in dr.all_records.items():
...     for rec in l_rec:
...         ann = dr.load_ann(rec)
...         d = ann["diagnosis"]["diagnosis_abbr"]
...         for item in d:
...             mat_cooccurance.loc[item,item] += 1
...         for i in range(len(d)-1):
...             for j in range(i+1,len(d)):
...                 mat_cooccurance.loc[d[i],d[j]] += 1
...                 mat_cooccurance.loc[d[j],d[i]] += 1

the diagonal entries are total occurrence of corresponding arrhythmias in the dataset
"""
