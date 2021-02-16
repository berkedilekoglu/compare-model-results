# comparison :camera:

comparison is a Python library to visualize results of different models.

> comparison just supports binary classification.

> To get visually qualified results, plot at most 7 models

> You can get detailed comparison for up to 3 models

### Example of Comparison Plot
Example
<kbd>
![Alt text](https://github.com/berkedilekoglu/compare-model-results/blob/main/plot_example.png)
</kbd>

## Installation

Download the repository to your local computer by using download button or git command.

```bash
git clone https://github.com/berkedilekoglu/compare-model-results.git
```

## Usage

```python
from comparison import *
# % matplotlib inline # Use it if you work in .ipynb 
comparison_device = comparison(X_val,y_val) #Create our class instance
Xx_val,yy_val = comparison_device.order_test_samples()  # Sort Gold Labels! 
#Use Xx_val and yy_val for prediction
```
If you take your predictions as a an array of probabilities. Use:
```python
comparison_device.set_prob_predictions("XGBoost Model",yy_pred,threshold=0.5)
# you can play with threhold. Labels will be determined by using threshold. Default is 0.5
```
If you take your predictions as an array of labels. Use:
```python
comparison_device.set_label_predictions("XGBoost Model",yy_pred) 
# This time there is no threshold because all predictions are already labeled!
```

To plot setted models:
```python
comparison_device.plot_predictions()
```
To take detailed report:
```python
comparison_device.plot_predictions()
```

Deletion:
```python
comparison_device.clear_all() #Clear all models with gold ones
comparison_device.delete_element(modelName) #Delete specified model
```

## Author
Berke Dilekoğlu

## License
[MIT](https://choosealicense.com/licenses/mit/)
