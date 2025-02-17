<div align="center">
	<h1>
		DEFCON
	</h1>
</div>

Source code for the paper **Better Prevent than Tackle: Valuing Defense in Soccer Based on Graph Neural Networks** by Kim et al., 2025 (under review).

## Introduction
**DEFCON (DEFensive CONtribution evaluator)** is a framework for evaluating the defensive contribution of soccer players in terms of reducing the expected return of the opposing team in a given situation.

![main](img/main.png)<br>


## Data Availability
Our code requires tracking data in the [kloppy](https://kloppy.pysport.org) format and event data in the [SPADL](https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html) format. However, the dataset used in this project is proprietary and cannot be publicly shared as it is an internal asset of the data provider. If you have your own dataset, you can use them after formatting.

## Training Component Models
The framework requires the estimation of six key probabilities for a given moment of a game as follows:
- **Expected return** indicating the difference between the goal-scoring and goal-conceding probabilities of the attacking team within the next 10 events (Section 3.2).
- **Receiver selection probability** that each teammate of the passer becomes the target (i.e., the "intended" receiver) of a pass (Section 3.3.1).
- **Pass success probability** that the pass is successfully sent to such intended receiver (Section 3.3.1).
- **Shot-blocking probability** that the shot made in the given situation would be blocked by a defender (Section 3.3.2).
- **Success-conditioned expected return** indicating the expected return of the attacking team under the condition that the pass to each teammate of the ball carrier were successful (Section 3.4).
- **Failure-conditioned intercept probability** that each of the opposing defenders would have caused the failure if a pass or a shot failed (Section 3.5).

Once you have formatted data, you can train the corresponding GNN models by executing the following scripts:
- Expected return: `sh scripts/scoring.sh`
- Pass success probability: `sh scripts/intent_success.sh`
- Receiver selection probability: `sh scripts/intent.sh`
- Shot-blocking probability: `sh scripts/shot_blocking.sh`
- Success-conditioned expected returns: `sh scripts/intent_scoring.sh`
- Failure-conditioned posterior probability: `sh scripts/failure_receiver.sh`

after installing the packages listed in `requirements.txt`. Besides, since our GNN models are based on [PyG](https://www.pyg.org), please install PyG whose version is compatible with your CUDA environment and PyTorch.

## Test and Visualization
For evaluating the model performance, please execute to `test.py` with the model ID that you want to evaluate. For example, if you want to evaluate the pass success model whose trial ID is 01, you can execute the following script:
```
python test.py --model_id intent_success/01
```

By executing cells in `plot.ipynb`, you can visualize the per-player defensive scores of a selected match as follows:
![score_plot](img/score_plot.png)<br>


