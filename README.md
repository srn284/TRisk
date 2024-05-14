# TRisk2
Repository for TRisk2 model
Manuscript: TRisk2 – a Transformer-based survival model for prediction of clinical outcomes in patients with heart failure: a multi-cohort study<br/>
Shishir Rao, Gholamreza Salimi-Khorshidi, Christopher Yau, Huimin Su, Nathalie Conrad, Mark Woodward, John GF Cleland, Kazem Rahimi
![Screenshot](TRisk2png.png)

How to use:<br/>
In "Demo" folder, run the "demoTRisk2.ipynb" file. A "forDemoTRisk2.parquet" file is provided to test/play and demonstrate how the vocabulary/year/age/etc function. The model has no pre-trained weights here but can fully run on the sampel (synthetic) cohort of 3000 patients.<br/>

The files in the "ModelPkg" folder contain model and data handling packages in addition to other necessary relevant files and helper functions.

Requirements:<br/>
torch >1.6.0<br/>
numpy 1.19.2<br/>
sklearn 0.23.2<br/>
pandas 1.1.3<br/>
<br/>
