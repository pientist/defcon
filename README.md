# DEFCON
Source code for KAIST DS535 term project **DEFCON: Leveraging GNNs to Measure Soccer Players' Defensive Performance** led by Hyunsung Kim.

### Introduction
**DEFCON (DEFensive CONtribution evaluator)** is a framework for evaluating the defensive contribution of players in terms of reducing the goal-scoring probability of the opposing team in a given situation.

### Data Availability
The dataset used in this project is proprietary and cannot be publicly shared as it is an internal asset of the data provider. For participants of the DS535 class, including the professor and TAs, please contact us via email to request access or further details regarding the data.

### Training Component Models
The framework requires the estimation of four key probabilities as follows:
- **Receiver selection probability** that each teammate of the passer becomes the target (i.e., the ``intended'' receiver) of a pass,
- **Pass success probability** that the pass is successfully sent to the intended receiver,
- **Success/Failure-conditioned goal probabilities** that the attacking team scores a goal in the near future if the pass to the intended receiver is successful/failed, and
- **Failure-conditioned intercept probability** that each of the opposing defenders intercept the ball provided the failure of the pass.

Once you have formatted data, you can train the models by executing the following scripts:
- `sh scripts/intent.sh`
- `sh scripts/intent_success.sh`
- `sh scripts/intent_scoring.sh`
- `sh scripts/failure_receiver.sh`

after installing the packages listed in `requirements.txt`. Besides, since our models are based on the PyG package, please install PyG whose version is compatible with your CUDA environment and PyTorch.

### Test and Visualization
For evaluating the model and visualizing the estimated probabilities, please refer to `test.ipynb`.


