# MFE-ACVP
A multi-modal ensemble learning tool for predicting anticoronavirus peptides, integrating sequence, structural, topological, and evolutionary features, and capable of generating new candidate peptides.
1. Datasets
Training sets: train_pos.fasta, train_neg.fasta
Test sets: test_pos.fasta, test_neg.fasta
Independent evaluation sets: independent_pos.fasta, independent_neg.fasta
All sequences are standard amino acid sequences (A, C, D, E, F, ...) in FASTA format.

2. Peptide Generation with GAN (MMaliGAN)
Use the improved MaliGAN model to generate new peptide candidates based on the positive training samples:
Goal: Generate a number of peptides equal to the positive training samples (1:1 ratio).
Command:
python Program/Model/GAN_Model.py
The generated peptides can be used to expand the training set and improve model generalization.

3. Structural Feature Generation
Generate structural features using external tools:
ESMATLAS
Function: Predicts 3D structures of peptides
Input: FASTA file
Output: .pdb file
Website: https://esmatlas.com/

NetSurfP-3.0
Function: Extracts secondary structure and conformational probabilities
Input: FASTA file
Output: .csv file
Website: https://services.healthtech.dtu.dk/services/NetSurfP-3.0/

4. Feature Extraction
Run the following scripts in order:
# Sequence features
python Program/Feature_extract/Seq_feature_extrac.py

# Structural features
python Program/Feature_extract/Structure_Feature_extrac.py

# Evolutionary features
python Program/Feature_extract/Evolu_feature_extrac.py

# Topological features
python Program/Feature_extract/Topological_feature_extrac.py

# Feature fusion and selection (final 100-dimensional vector)
python Program/Feature_extract/Feature_fusion_selection.py

5. Ensemble Learning Prediction
Train and evaluate the multi-modal ensemble model
Note: Make sure all base models (RF, XGBoost, CatBoost...) are trained beforehand, as the ensemble prediction depends on them.
python Program/Model/Ensemble.py
