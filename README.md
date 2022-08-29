# OpticReader-v6.1
Entry point in main.py
## Required packages
- Opencv: pip install opencv-python
- numpy: pip install numpy
- joblib: pip install joblib
- pyzbar: pip install pyzbar
- xlsxwriter: pip install xlsxwriter
- dearpygui: pip install dearpygui

Probably sklearn (pip install sklearn) is required for SCV prediction

## File description
- **custom_utils.py**: general purpose function container (pyzbar is required, numpy is use only in type annotation);
- **evaluator.py**: heavy lifter for answer recogniction  (openCV and joblib are required. Additional libraries from the std ones are imported);
- **main.py**: entry point, it handles the GUI and calls evaluator form evaluator.py;
-  **reduced.joblib**: SVC model trained with separeted scripts;
-  **risposte.txt**: list of all the corrected answers for a given test. DO NOT MODIFY IT BY HAND, launch main.py and edit them from there.

## To-Dos
-  Train again the model with normally drawn answers;
-  In "evaluator.py" the function find_n_black_point_on_row is fuctional but not optimized
-  Remove the current align system and replace it with warpAffine function from openCV. Standard blank sheets will be modified after this
