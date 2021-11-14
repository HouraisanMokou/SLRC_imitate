# structure
the **data** folder is for original data and corpus

the **result** folder is for saving the model

the **src** folder is for source code

the **scipt** folder is for shell script

the **log** folder to place logging


***
## src:
**main.py:** controlling program

**process.py:** preprocess the data to format of us

***
### src.common
**metrics:** the function related with statistics, like 

**utils:** convert the data form, when needed or 

**constants:** some constants

**runner** to control the running model

**reader** statics the data after preprocess(something such as total number of users), like a DTO layer.

the meaning of the runner of reader is that to run different model with same method, so that the code is less.

**expand:** the folder for expanded layer, function for the pytorch, if needed.

***
### src.models
the folder of models

**SLRC:** the class for base model, all the runnable method inherited this class

**SLRC_XX** the son of **SLRC** XX is a model name

