# TRisk
Repository for TRisk model<br/>
Manuscripts (in preparation) associated: <br/>
Refined selection of individuals for preventive cardiovascular disease treatment with a Transformer-based risk model<br/>
Shishir Rao, Yikuan Li, Mohammad Mamouei, Gholamreza Salimi-Khorshidi, Malgorzata Wamil, Milad Nazarzadeh, Christopher Yau, Gary S Collins, Rod Jackson, Andrew Vickers, Goodarz Danaei, Kazem Rahimi.<br/><br/>

A Transformer-based survival model for point-of-care prediction of all-cause mortality in heart failure patients: a multi-cohort study<br/>
Shishir Rao, Gholamreza Salimi-Khorshidi, Christopher Yau, Huimin Su, Nathalie Conrad, Mark Woodward, John GF Cleland, Kazem Rahimi<br/>
![Screenshot](triskmodel.png)

How to use:<br/>
In "Demo" folder, run the "demoTRisk.ipynb" file. A "forDemoTRisk2.parquet" file is provided to test/play and demonstrate how the vocabulary/year/age/etc function. The model has no pre-trained weights here but can fully run on the sampel (synthetic) cohort of 3000 patients. The files in the "ModelPkg" folder contain model and data handling packages in addition to other necessary relevant files and helper functions.<br/>

Additionally, all diagnostic codes used to identify secondary outcomes in the UK validation study is provided in "DiagCodes" folder within the phenotyping document, "DiagCodes.docx".

Requirements:<br/>
torch >1.6.0<br/>
numpy 1.19.2<br/>
sklearn 0.23.2<br/>
pandas 1.1.3<br/>
<br/>
